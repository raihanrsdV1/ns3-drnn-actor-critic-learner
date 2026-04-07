/*
 * wifi-sim.cc — WiFi Multi-Flow Parking-Lot TCP Simulation
 *
 * Topology (beyond dumbbell — dual bottleneck + WiFi contention + cross-traffic):
 *
 *              WiFi 802.11n                   Wired Parking-Lot
 *   STA0 ─┐                                                          ┌── Server
 *   STA1 ─┤── AP ──[100Mbps/1ms]── R1 ──[8Mbps/10ms/60p]── R2 ──[4Mbps/25ms/40p]──┘
 *   STA2 ─┘                         │
 *                              CrossSrc (UDP OnOff ~1 Mbps avg)
 *
 *   Dynamics beyond simple dumbbell:
 *     1. WiFi CSMA/CA contention among 3 STAs (shared medium, variable airtime)
 *     2. Two bottleneck links in series (8 Mbps + 4 Mbps) — "parking lot"
 *     3. Bursty UDP cross-traffic competing with TCP at R1→R2
 *     4. 3 TCP flows competing for bottleneck bandwidth → fairness effects
 *     5. MinstrelHt WiFi rate adaptation → variable wireless capacity
 *
 *   Queue sizing (guarantees drops):
 *     Link2 (main bottleneck): 4 Mbps, ~80 ms RTT, BDP≈40 KB/flow
 *       → 3 flows × 40 KB = 120 KB demand >> 40p×1448 = 57.9 KB queue → DROPS
 *     Link1: 8 Mbps, ~80 ms RTT, 60p queue → drops when UDP bursts coincide
 *
 * Usage:
 *   ./ns3 run wifi-sim -- --transport=cubic
 *   ./ns3 run wifi-sim -- --transport=reno
 *   python3 scratch/wifi-agent.py --port=5557 &
 *   ./ns3 run wifi-sim -- --transport=drnn --port=5557
 *
 * CSV output (written to ns-3.40/ working directory):
 *   w_{transport}_throughput.csv — Time, Total, Flow0..2, Drops_L1, Drops_L2
 *   w_{transport}_cwnd.csv      — Time, Flow0_CWND..Flow2_CWND  (bytes)
 *   w_{transport}_rtt.csv       — Time, Flow0_RTT..Flow2_RTT    (ms)
 *
 * DRNN OpenGym state (9 floats, 3 flows × [cwnd_bytes, rtt_ms, bytesInFlight]):
 *   Updated every 100 ms (dt=0.1 s).
 *   Action space: 1 continuous float in [DRNN_MIN_CWND, DRNN_MAX_CWND] (bytes).
 *   The agent outputs a direct target cwnd; IncreaseWindow sets it immediately.
 *   This is the continuous-control variant of wifi-sim.cc.
 */

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/wifi-module.h"
#include "ns3/mobility-module.h"
#include "ns3/applications-module.h"
#include "ns3/traffic-control-module.h"
#include "ns3/opengym-module.h"
#include <iostream>
#include <fstream>
#include <sstream>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("WifiSim");

// ═══════════════════════════════════════════════════════════════════
// CUSTOM CC: TcpWifiDrnn  (CONTINUOUS-CONTROL variant)
//
// The Python LSTM-Actor-Critic agent outputs ONE float per step:
//   g_target_cwnd  ∈  [DRNN_MIN_CWND, DRNN_MAX_CWND]  (bytes)
//
// IncreaseWindow() just clamps that value and assigns it directly
// to tcb->m_cWnd — no AIMD arithmetic, no multipliers.
//
// On loss: standard cwnd/2 safety halving (agent learned to avoid
// queuing before this fires).
// ═══════════════════════════════════════════════════════════════════
class TcpWifiDrnn : public TcpCongestionOps
{
public:
  static TypeId GetTypeId (void);
  TcpWifiDrnn ();
  TcpWifiDrnn (const TcpWifiDrnn &sock);
  ~TcpWifiDrnn () override;

  std::string GetName () const override { return "TcpWifiDrnn"; }

  uint32_t GetSsThresh (Ptr<const TcpSocketState> tcb,
                        uint32_t bytesInFlight) override;
  void IncreaseWindow (Ptr<TcpSocketState> tcb,
                       uint32_t segmentsAcked) override;
  Ptr<TcpCongestionOps> Fork () override;

  // Set by OpenGym ExecuteAction each step — direct target cwnd in bytes
  static float    g_target_cwnd;
};

// Defaults: start at 1× BDP per flow (≈ 12 KB); full range 2 seg – 144 KB
static const uint32_t DRNN_MIN_CWND =  2 * 1448;       //  2 segments
static const uint32_t DRNN_MAX_CWND = 144 * 1024;      // 144 KB  (4× BDP)

float TcpWifiDrnn::g_target_cwnd = (float)(12 * 1024); // 12 KB initial

NS_OBJECT_ENSURE_REGISTERED (TcpWifiDrnn);

TypeId
TcpWifiDrnn::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::TcpWifiDrnn")
    .SetParent<TcpCongestionOps> ()
    .SetGroupName ("Internet")
    .AddConstructor<TcpWifiDrnn> ();
  return tid;
}

TcpWifiDrnn::TcpWifiDrnn ()  : TcpCongestionOps () {}
TcpWifiDrnn::TcpWifiDrnn (const TcpWifiDrnn &s) : TcpCongestionOps (s) {}
TcpWifiDrnn::~TcpWifiDrnn () {}

uint32_t
TcpWifiDrnn::GetSsThresh (Ptr<const TcpSocketState> tcb,
                           uint32_t /* bytesInFlight */)
{
  // On packet loss, halve cwnd — standard AIMD safety net.
  // The agent should have already reduced cwnd via direct assignment
  // before this fires, but if a loss slips through this caps damage.
  return std::max<uint32_t> (DRNN_MIN_CWND,
                              tcb->m_cWnd.Get () / 2);
}

void
TcpWifiDrnn::IncreaseWindow (Ptr<TcpSocketState> tcb, uint32_t segmentsAcked)
{
  if (segmentsAcked == 0)
    return;

  // Direct assignment: clamp the agent's target to [MIN, MAX].
  // No AIMD arithmetic — the agent owns cwnd completely.
  uint32_t seg    = tcb->m_segmentSize;
  uint32_t target = static_cast<uint32_t> (
      std::max ((float)DRNN_MIN_CWND,
      std::min ((float)DRNN_MAX_CWND, g_target_cwnd)));
  // Align to segment boundary to avoid tiny mismatches
  target = std::max (2 * seg, (target / seg) * seg);
  tcb->m_cWnd = target;
}

Ptr<TcpCongestionOps>
TcpWifiDrnn::Fork ()
{
  return CopyObject<TcpWifiDrnn> (this);
}

// ═══════════════════════════ constants ═══════════════════════════
static const uint32_t NUM_FLOWS = 3;

// ═══════════════════════════ per-flow state ═══════════════════════════
struct FlowState
{
  uint64_t lastTotalRx  = 0;
  uint32_t cwnd         = 0;
  double   rttMs        = 0.0;
  uint32_t bytesInFlight = 0;
};

static FlowState       g_flow[NUM_FLOWS];
static Ptr<PacketSink> g_sink[NUM_FLOWS];
static uint32_t        g_staNodeId[NUM_FLOWS];

// ═══════════════════════════ drop counters ═══════════════════════════
static uint32_t g_dropsL1     = 0;   // R1→R2 queue drops
static uint32_t g_dropsL2     = 0;   // R2→Server queue drops
static uint32_t g_lastDropsL1 = 0;
static uint32_t g_lastDropsL2 = 0;

// Separate counters for gym-observation drop deltas
static uint32_t g_gymLastDropsL1 = 0;
static uint32_t g_gymLastDropsL2 = 0;

// WiFi-layer drop counters
static uint32_t g_wifiMacDrops       = 0;  // MPDU dropped after max retries
static uint32_t g_wifiPhyTxDrops     = 0;  // PHY TX drops (all WiFi nodes)
static uint32_t g_wifiPhyRxDrops     = 0;  // PHY RX drops (all WiFi nodes)
static uint32_t g_lastWifiMacDrops   = 0;
static uint32_t g_lastWifiPhyTxDrops = 0;
static uint32_t g_lastWifiPhyRxDrops = 0;
static uint32_t g_gymLastWifiMacDrops = 0;

// ═══════════════════════════ CSV streams ═══════════════════════════
static std::ofstream g_tputFile;
static std::ofstream g_cwndFile;
static std::ofstream g_rttFile;

static double   g_simTime  = 60.0;
static bool     g_isDrnn   = false;

// ═══════════════════════════ OpenGym ═══════════════════════════
static Ptr<OpenGymInterface> g_openGym;
static uint32_t g_gymPort = 5557;
// Observation: 12 floats — per flow [cwnd, rtt, bif] × 3 + [dropsL1, dropsL2, wifiMacDrops]
static const uint32_t OBS_DIM = NUM_FLOWS * 3 + 3;

// ══════════════════════════ trace callbacks ══════════════════════════

static void
CwndTrace (uint32_t flowId, uint32_t /* oldVal */, uint32_t newVal)
{
  g_flow[flowId].cwnd = newVal;
}

static void
RttTrace (uint32_t flowId, Time /* oldVal */, Time newVal)
{
  g_flow[flowId].rttMs = newVal.GetMilliSeconds ();
}

static void
InFlightTrace (uint32_t flowId, uint32_t /* oldVal */, uint32_t newVal)
{
  g_flow[flowId].bytesInFlight = newVal;
}

static void
DropTraceL1 (Ptr<const QueueDiscItem> /* item */)
{
  ++g_dropsL1;
}

static void
DropTraceL2 (Ptr<const QueueDiscItem> /* item */)
{
  ++g_dropsL2;
}

// WiFi MAC drop: MPDU discarded after exhausting max retransmissions
static void
WifiMacDropTrace (std::string /* ctx */,
                  Ptr<const WifiMpdu> /* mpdu */,
                  WifiMacDropReason /* reason */)
{
  ++g_wifiMacDrops;
}

// WiFi PHY TX drop (e.g. channel busy, internal TX failure)
static void
WifiPhyTxDropTrace (Ptr<const Packet> /* p */)
{
  ++g_wifiPhyTxDrops;
}

// WiFi PHY RX drop (e.g. collision, low SNR)
static void
WifiPhyRxDropTrace (Ptr<const Packet> /* p */, WifiPhyRxfailureReason /* reason */)
{
  ++g_wifiPhyRxDrops;
}

// Connect WiFi drop traces on all STAs and the AP
static void
ConnectWifiDropTraces (NodeContainer staNodes, NodeContainer apNode)
{
  // Combine all WiFi nodes for PHY traces
  NodeContainer wifiNodes;
  wifiNodes.Add (staNodes);
  wifiNodes.Add (apNode);

  for (uint32_t i = 0; i < wifiNodes.GetN (); ++i)
    {
      std::ostringstream phyTxPath, phyRxPath, macPath;
      uint32_t nid = wifiNodes.Get (i)->GetId ();
      phyTxPath << "/NodeList/" << nid
                << "/DeviceList/0/$ns3::WifiNetDevice/Phy/PhyTxDrop";
      phyRxPath << "/NodeList/" << nid
                << "/DeviceList/0/$ns3::WifiNetDevice/Phy/PhyRxDrop";
      macPath   << "/NodeList/" << nid
                << "/DeviceList/0/$ns3::WifiNetDevice/Mac/DroppedMpdu";

      Config::Connect (phyTxPath.str (), MakeCallback (&WifiPhyTxDropTrace));
      Config::Connect (phyRxPath.str (), MakeCallback (&WifiPhyRxDropTrace));
      Config::Connect (macPath.str (),   MakeCallback (&WifiMacDropTrace));
    }
  NS_LOG_UNCOND ("WiFi drop traces connected for "
                 << wifiNodes.GetN () << " nodes");
}

// ══════════════════════════ OpenGym callbacks ══════════════════════════

Ptr<OpenGymSpace>
GetObsSpace ()
{
  // 12 values: [cwnd0,rtt0,bif0, cwnd1,rtt1,bif1, cwnd2,rtt2,bif2, dropsL1, dropsL2, wifiMacDrops]
  std::vector<uint32_t> shape = {(uint32_t)OBS_DIM};
  Ptr<OpenGymBoxSpace> space = CreateObject<OpenGymBoxSpace> (
      0.0f, 1e9f, shape, TypeNameGet<float> ());
  return space;
}

Ptr<OpenGymSpace>
GetActSpace ()
{
  // Continuous action: one float — target cwnd in bytes for all flows.
  // Range: [DRNN_MIN_CWND, DRNN_MAX_CWND].
  std::vector<uint32_t> shape = {1};
  Ptr<OpenGymBoxSpace> space = CreateObject<OpenGymBoxSpace> (
      (float)DRNN_MIN_CWND, (float)DRNN_MAX_CWND,
      shape, TypeNameGet<float> ());
  return space;
}

Ptr<OpenGymDataContainer>
GetObs ()
{
  Ptr<OpenGymBoxContainer<float>> box =
    CreateObject<OpenGymBoxContainer<float>> ();
  for (uint32_t i = 0; i < NUM_FLOWS; ++i)
    {
      box->AddValue ((float)g_flow[i].cwnd);
      box->AddValue ((float)g_flow[i].rttMs);
      box->AddValue ((float)g_flow[i].bytesInFlight);
    }
  // Append per-step drop deltas — direct packet-loss signal for the agent
  uint32_t dL1   = g_dropsL1      - g_gymLastDropsL1;
  uint32_t dL2   = g_dropsL2      - g_gymLastDropsL2;
  uint32_t dWifi = g_wifiMacDrops - g_gymLastWifiMacDrops;
  g_gymLastDropsL1      = g_dropsL1;
  g_gymLastDropsL2      = g_dropsL2;
  g_gymLastWifiMacDrops = g_wifiMacDrops;
  box->AddValue ((float)dL1);
  box->AddValue ((float)dL2);
  box->AddValue ((float)dWifi);
  return box;
}

float
GetReward ()
{
  // Throughput component: sum cwnd/rtt_sec across active flows, normalise by 4 Mbps
  double totalTput = 0.0;
  int    active    = 0;
  double avgRtt    = 0.0;
  for (uint32_t i = 0; i < NUM_FLOWS; ++i)
    {
      if (g_flow[i].cwnd > 0 && g_flow[i].rttMs > 0)
        {
          double rtt_s = g_flow[i].rttMs / 1000.0;
          totalTput += (double)g_flow[i].cwnd / rtt_s * 8.0 / 1e6; // Mbps
          avgRtt    += g_flow[i].rttMs;
          ++active;
        }
    }
  if (active == 0)
    return 0.0f;
  avgRtt /= active;

  double util     = std::min (totalTput / 4.0, 1.0);       // cap at 4 Mbps bottleneck
  double latPen   = 0.8 * std::max (0.0, avgRtt / 72.0 - 1.0); // min RTT ~72 ms
  // Fairness (Jain's index)
  double sumT = 0.0, sumT2 = 0.0;
  for (uint32_t i = 0; i < NUM_FLOWS; ++i)
    {
      if (g_flow[i].cwnd > 0 && g_flow[i].rttMs > 0)
        {
          double t = (double)g_flow[i].cwnd / (g_flow[i].rttMs / 1000.0) * 8.0 / 1e6;
          sumT += t; sumT2 += t * t;
        }
    }
  double jain = (sumT2 > 0) ? (sumT * sumT) / (active * sumT2) : 1.0;
  float  r    = (float)(util - latPen + 0.3 * jain);
  return r;
}

bool
GetDone ()
{
  return (Simulator::Now ().GetSeconds () >= g_simTime);
}

std::string
GetInfo ()
{
  return "";
}

bool
ExecuteAction (Ptr<OpenGymDataContainer> action)
{
  // The Python agent sends a Box container with one float: target cwnd.
  Ptr<OpenGymBoxContainer<float>> box =
    DynamicCast<OpenGymBoxContainer<float>> (action);
  if (box && box->GetData ().size () >= 1)
    {
      float val = box->GetData ()[0];
      // Simple clamp — the Python agent already uses a tight [4KB, 24KB]
      // range so no additional smoothing is needed.
      TcpWifiDrnn::g_target_cwnd =
        std::max ((float)DRNN_MIN_CWND,
        std::min ((float)DRNN_MAX_CWND, val));
    }
  return true;
}

static void
DrnnStep (double dt)
{
  g_openGym->NotifyCurrentState ();
  if (!GetDone ())
    Simulator::Schedule (Seconds (dt), &DrnnStep, dt);
}

// ══════════════════════ periodic measurement ══════════════════════

static void
CalcThroughput ()
{
  double now = Simulator::Now ().GetSeconds ();
  if (now >= g_simTime)
    return;

  double flowTput[NUM_FLOWS];
  double total = 0.0;

  for (uint32_t i = 0; i < NUM_FLOWS; ++i)
    {
      uint64_t rx  = g_sink[i]->GetTotalRx ();
      double mbps  = (rx - g_flow[i].lastTotalRx) * 8.0 / 0.1 / 1e6;
      g_flow[i].lastTotalRx = rx;
      flowTput[i] = mbps;
      total += mbps;
    }

  uint32_t dL1      = g_dropsL1      - g_lastDropsL1;
  g_lastDropsL1     = g_dropsL1;
  uint32_t dL2      = g_dropsL2      - g_lastDropsL2;
  g_lastDropsL2     = g_dropsL2;
  uint32_t dWifiMac = g_wifiMacDrops  - g_lastWifiMacDrops;
  g_lastWifiMacDrops = g_wifiMacDrops;
  uint32_t dWifiPhy = (g_wifiPhyTxDrops - g_lastWifiPhyTxDrops)
                    + (g_wifiPhyRxDrops  - g_lastWifiPhyRxDrops);
  g_lastWifiPhyTxDrops = g_wifiPhyTxDrops;
  g_lastWifiPhyRxDrops = g_wifiPhyRxDrops;

  // throughput CSV
  g_tputFile << now << "," << total;
  for (uint32_t i = 0; i < NUM_FLOWS; ++i)
    g_tputFile << "," << flowTput[i];
  g_tputFile << "," << dL1 << "," << dL2
             << "," << dWifiMac << "," << dWifiPhy << "\n";

  // cwnd CSV (snapshot current value)
  g_cwndFile << now;
  for (uint32_t i = 0; i < NUM_FLOWS; ++i)
    g_cwndFile << "," << g_flow[i].cwnd;
  g_cwndFile << "\n";

  // rtt CSV (snapshot current value)
  g_rttFile << now;
  for (uint32_t i = 0; i < NUM_FLOWS; ++i)
    g_rttFile << "," << g_flow[i].rttMs;
  g_rttFile << "\n";

  Simulator::Schedule (Seconds (0.1), &CalcThroughput);
}

// baseline Rx counts so first CalcThroughput interval is accurate
static void
StartMeasurement ()
{
  for (uint32_t i = 0; i < NUM_FLOWS; ++i)
    g_flow[i].lastTotalRx = g_sink[i]->GetTotalRx ();
  Simulator::Schedule (Seconds (0.1), &CalcThroughput);
}

// ══════════════ connect per-socket CWND/RTT/InFlight traces ══════════════

static void
ConnectSocketTraces ()
{
  for (uint32_t i = 0; i < NUM_FLOWS; ++i)
    {
      std::ostringstream cwndPath, rttPath, ifPath;
      cwndPath << "/NodeList/" << g_staNodeId[i]
               << "/$ns3::TcpL4Protocol/SocketList/0/CongestionWindow";
      rttPath  << "/NodeList/" << g_staNodeId[i]
               << "/$ns3::TcpL4Protocol/SocketList/0/RTT";
      ifPath   << "/NodeList/" << g_staNodeId[i]
               << "/$ns3::TcpL4Protocol/SocketList/0/BytesInFlight";

      Config::ConnectWithoutContext (cwndPath.str (),
                                    MakeBoundCallback (&CwndTrace, i));
      Config::ConnectWithoutContext (rttPath.str (),
                                    MakeBoundCallback (&RttTrace, i));
      Config::ConnectWithoutContext (ifPath.str (),
                                    MakeBoundCallback (&InFlightTrace, i));
    }
  NS_LOG_UNCOND ("Socket traces connected for " << NUM_FLOWS << " flows");
}

// ═══════════════════════════════════════════════════════════════════
//  main
// ═══════════════════════════════════════════════════════════════════

int
main (int argc, char *argv[])
{
  std::string transport = "cubic";

  CommandLine cmd (__FILE__);
  cmd.AddValue ("transport", "TCP variant: cubic | reno | drnn", transport);
  cmd.AddValue ("simTime",   "Simulation duration (s)",          g_simTime);
  cmd.AddValue ("port",      "OpenGym port (drnn only)",         g_gymPort);
  cmd.Parse (argc, argv);

  g_isDrnn = (transport == "drnn");

  // ────────── TCP variant ──────────
  if (transport == "cubic")
    Config::SetDefault ("ns3::TcpL4Protocol::SocketType",
                        TypeIdValue (TcpCubic::GetTypeId ()));
  else if (transport == "reno")
    Config::SetDefault ("ns3::TcpL4Protocol::SocketType",
                        TypeIdValue (TcpNewReno::GetTypeId ()));
  else if (transport == "drnn")
    Config::SetDefault ("ns3::TcpL4Protocol::SocketType",
                        TypeIdValue (TcpWifiDrnn::GetTypeId ()));
  else
    NS_FATAL_ERROR ("Unknown transport: " << transport);

  // ────────── TCP knobs ──────────
  Config::SetDefault ("ns3::TcpSocket::SegmentSize",  UintegerValue (1448));
  Config::SetDefault ("ns3::TcpSocket::SndBufSize",   UintegerValue (1 << 20)); // 1 MB
  Config::SetDefault ("ns3::TcpSocket::RcvBufSize",   UintegerValue (1 << 20)); // 1 MB
  // Large initial ssThresh: stay in slow-start until first real loss
  Config::SetDefault ("ns3::TcpSocket::InitialSlowStartThreshold",
                      UintegerValue (10 * 1024 * 1024)); // 10 MB ≈ infinite

  // Device queue 1p so FifoQueueDisc is the sole buffering point
  Config::SetDefault ("ns3::DropTailQueue<Packet>::MaxSize",
                      QueueSizeValue (QueueSize ("1p")));

  // ═══════════════════════ nodes ═══════════════════════
  NodeContainer staNodes;    staNodes.Create (NUM_FLOWS);   // WiFi clients
  NodeContainer apNode;      apNode.Create (1);             // WiFi AP
  NodeContainer r1Node;      r1Node.Create (1);             // Router 1
  NodeContainer r2Node;      r2Node.Create (1);             // Router 2
  NodeContainer serverNode;  serverNode.Create (1);         // Wired server
  NodeContainer crossNode;   crossNode.Create (1);          // UDP cross-traffic src

  for (uint32_t i = 0; i < NUM_FLOWS; ++i)
    g_staNodeId[i] = staNodes.Get (i)->GetId ();

  // ═══════════════════════ WiFi 802.11n ═══════════════════════
  YansWifiChannelHelper wifiChannel = YansWifiChannelHelper::Default ();
  YansWifiPhyHelper     wifiPhy;
  wifiPhy.SetChannel (wifiChannel.Create ());

  WifiHelper wifi;
  wifi.SetStandard (WIFI_STANDARD_80211n);
  wifi.SetRemoteStationManager ("ns3::MinstrelHtWifiManager");

  WifiMacHelper mac;
  Ssid ssid = Ssid ("wifi-sim");

  // STA devices
  mac.SetType ("ns3::StaWifiMac",
               "Ssid", SsidValue (ssid),
               "ActiveProbing", BooleanValue (false));
  NetDeviceContainer staDevices = wifi.Install (wifiPhy, mac, staNodes);

  // AP device
  mac.SetType ("ns3::ApWifiMac", "Ssid", SsidValue (ssid));
  NetDeviceContainer apDevice = wifi.Install (wifiPhy, mac, apNode);

  // ────────── mobility (fixed positions, close to AP) ──────────
  MobilityHelper mobility;
  Ptr<ListPositionAllocator> posAlloc = CreateObject<ListPositionAllocator> ();
  posAlloc->Add (Vector ( 0.0, 0.0, 0.0));  // STA0
  posAlloc->Add (Vector ( 5.0, 0.0, 0.0));  // STA1
  posAlloc->Add (Vector (10.0, 0.0, 0.0));  // STA2
  posAlloc->Add (Vector ( 5.0, 5.0, 0.0));  // AP  (close — good SNR)
  mobility.SetPositionAllocator (posAlloc);
  mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  mobility.Install (staNodes);
  mobility.Install (apNode);

  // Wired nodes don't need mobility, but ns-3 WiFi expects it set on
  // all nodes in the same simulation — safe to skip for non-WiFi nodes.

  // ═══════════════════════ wired links ═══════════════════════

  // AP ↔ R1 : fast feeder link
  PointToPointHelper p2pFast;
  p2pFast.SetDeviceAttribute  ("DataRate", StringValue ("100Mbps"));
  p2pFast.SetChannelAttribute ("Delay",    StringValue ("1ms"));
  NetDeviceContainer apR1Dev = p2pFast.Install (apNode.Get (0), r1Node.Get (0));

  // CrossSrc ↔ R1 : fast link for UDP injector
  NetDeviceContainer crossR1Dev = p2pFast.Install (crossNode.Get (0), r1Node.Get (0));

  // R1 ↔ R2 : BOTTLENECK LINK 1 (8 Mbps, 10 ms, 60p queue)
  PointToPointHelper p2pBn1;
  p2pBn1.SetDeviceAttribute  ("DataRate", StringValue ("8Mbps"));
  p2pBn1.SetChannelAttribute ("Delay",    StringValue ("10ms"));
  NetDeviceContainer bn1Dev = p2pBn1.Install (r1Node.Get (0), r2Node.Get (0));

  // R2 ↔ Server : BOTTLENECK LINK 2 — main drop point (4 Mbps, 25 ms, 40p queue)
  PointToPointHelper p2pBn2;
  p2pBn2.SetDeviceAttribute  ("DataRate", StringValue ("4Mbps"));
  p2pBn2.SetChannelAttribute ("Delay",    StringValue ("25ms"));
  NetDeviceContainer bn2Dev = p2pBn2.Install (r2Node.Get (0), serverNode.Get (0));

  // ═══════════════════════ internet stack ═══════════════════════
  InternetStackHelper internet;
  internet.Install (staNodes);
  internet.Install (apNode);
  internet.Install (r1Node);
  internet.Install (r2Node);
  internet.Install (serverNode);
  internet.Install (crossNode);

  // ═══════════════════════ traffic control ═══════════════════════
  // Install BEFORE Ipv4AddressHelper::Assign() to override default pfifo_fast

  // Link 1: R1→R2 direction (60 packets ≈ 86.9 KB)
  TrafficControlHelper tch1;
  tch1.SetRootQueueDisc ("ns3::FifoQueueDisc",
                         "MaxSize", StringValue ("60p"));
  QueueDiscContainer qd1 = tch1.Install (bn1Dev.Get (0));
  qd1.Get (0)->TraceConnectWithoutContext ("Drop",
                                           MakeCallback (&DropTraceL1));

  // Link 2: R2→Server direction (40 packets ≈ 57.9 KB) — primary drop point
  TrafficControlHelper tch2;
  tch2.SetRootQueueDisc ("ns3::FifoQueueDisc",
                         "MaxSize", StringValue ("40p"));
  QueueDiscContainer qd2 = tch2.Install (bn2Dev.Get (0));
  qd2.Get (0)->TraceConnectWithoutContext ("Drop",
                                           MakeCallback (&DropTraceL2));

  // ═══════════════════════ IP addresses ═══════════════════════
  Ipv4AddressHelper ipv4;

  // WiFi subnet  10.1.1.0/24
  ipv4.SetBase ("10.1.1.0", "255.255.255.0");
  ipv4.Assign (staDevices);
  ipv4.Assign (apDevice);

  // AP ↔ R1  10.1.2.0/24
  ipv4.SetBase ("10.1.2.0", "255.255.255.0");
  ipv4.Assign (apR1Dev);

  // CrossSrc ↔ R1  10.1.3.0/24
  ipv4.SetBase ("10.1.3.0", "255.255.255.0");
  ipv4.Assign (crossR1Dev);

  // R1 ↔ R2  10.1.4.0/24
  ipv4.SetBase ("10.1.4.0", "255.255.255.0");
  ipv4.Assign (bn1Dev);

  // R2 ↔ Server  10.1.5.0/24
  ipv4.SetBase ("10.1.5.0", "255.255.255.0");
  Ipv4InterfaceContainer serverIface = ipv4.Assign (bn2Dev);

  Ipv4GlobalRoutingHelper::PopulateRoutingTables ();
  Ipv4Address serverAddr = serverIface.GetAddress (1);  // Server's IP

  // ═══════════════════════ TCP applications ═══════════════════════
  uint16_t basePort = 9000;

  for (uint32_t i = 0; i < NUM_FLOWS; ++i)
    {
      uint16_t port = basePort + i;

      // PacketSink on Server
      PacketSinkHelper sinkHelper ("ns3::TcpSocketFactory",
          InetSocketAddress (Ipv4Address::GetAny (), port));
      ApplicationContainer sinkApp = sinkHelper.Install (serverNode.Get (0));
      sinkApp.Start (Seconds (0.0));
      sinkApp.Stop  (Seconds (g_simTime));
      g_sink[i] = DynamicCast<PacketSink> (sinkApp.Get (0));

      // BulkSend from STA (staggered: 0.5 s, 0.7 s, 0.9 s)
      BulkSendHelper srcHelper ("ns3::TcpSocketFactory",
          InetSocketAddress (serverAddr, port));
      srcHelper.SetAttribute ("MaxBytes", UintegerValue (0));   // unlimited
      srcHelper.SetAttribute ("SendSize", UintegerValue (1448));
      ApplicationContainer srcApp = srcHelper.Install (staNodes.Get (i));
      srcApp.Start (Seconds (0.5 + i * 0.2));
      srcApp.Stop  (Seconds (g_simTime));
    }

  // ═══════════════════════ UDP cross-traffic ═══════════════════════
  // Bursty ON/OFF: 2 Mbps during ON (0.5 s), off 0.5 s → avg ~1 Mbps
  // Joins at R1, competes with TCP on Link1 (8 Mbps) and Link2 (4 Mbps)

  uint16_t udpPort = 10000;

  PacketSinkHelper udpSinkHelper ("ns3::UdpSocketFactory",
      InetSocketAddress (Ipv4Address::GetAny (), udpPort));
  ApplicationContainer udpSinkApp = udpSinkHelper.Install (serverNode.Get (0));
  udpSinkApp.Start (Seconds (0.0));
  udpSinkApp.Stop  (Seconds (g_simTime));

  OnOffHelper onoff ("ns3::UdpSocketFactory",
                     InetSocketAddress (serverAddr, udpPort));
  onoff.SetAttribute ("DataRate",   StringValue ("2Mbps"));
  onoff.SetAttribute ("PacketSize", UintegerValue (512));
  onoff.SetAttribute ("OnTime",  StringValue (
      "ns3::ConstantRandomVariable[Constant=0.5]"));
  onoff.SetAttribute ("OffTime", StringValue (
      "ns3::ConstantRandomVariable[Constant=0.5]"));

  ApplicationContainer crossApp = onoff.Install (crossNode.Get (0));
  crossApp.Start (Seconds (2.0));   // starts after TCP flows establish
  crossApp.Stop  (Seconds (g_simTime));

  // ═══════════════════════ open CSV files ═══════════════════════
  std::string pfx = "w_" + transport + "_";

  g_tputFile.open (pfx + "throughput.csv");
  g_tputFile << "Time,Total";
  for (uint32_t i = 0; i < NUM_FLOWS; ++i)
    g_tputFile << ",Flow" << i;
  g_tputFile << ",Drops_L1,Drops_L2,Drops_WiFiMAC,Drops_WiFiPHY\n";

  g_cwndFile.open (pfx + "cwnd.csv");
  g_cwndFile << "Time";
  for (uint32_t i = 0; i < NUM_FLOWS; ++i)
    g_cwndFile << ",Flow" << i << "_CWND";
  g_cwndFile << "\n";

  g_rttFile.open (pfx + "rtt.csv");
  g_rttFile << "Time";
  for (uint32_t i = 0; i < NUM_FLOWS; ++i)
    g_rttFile << ",Flow" << i << "_RTT";
  g_rttFile << "\n";

  // ═══════════════════════ WiFi drop traces ═══════════════════════
  // WiFi devices are fully configured after Simulator::Run() begins,
  // so schedule the hook-up slightly after t=0.
  Simulator::Schedule (Seconds (0.1), &ConnectWifiDropTraces, staNodes, apNode);

  // ═══════════════════════ schedule traces ═══════════════════════
  // BulkSend sockets are created at Start time (0.5–0.9 s).
  // Connect traces at 1.5 s — well after all sockets exist.
  Simulator::Schedule (Seconds (1.5), &ConnectSocketTraces);
  // Baseline Rx counts at 1.6 s, first CalcThroughput at 1.7 s
  Simulator::Schedule (Seconds (1.6), &StartMeasurement);

  // ═══════════════════════ OpenGym (drnn only) ═══════════════════════
  if (g_isDrnn)
    {
      g_openGym = CreateObject<OpenGymInterface> (g_gymPort);
      g_openGym->SetGetObservationSpaceCb  (MakeCallback (&GetObsSpace));
      g_openGym->SetGetActionSpaceCb       (MakeCallback (&GetActSpace));
      g_openGym->SetGetObservationCb       (MakeCallback (&GetObs));
      g_openGym->SetGetRewardCb            (MakeCallback (&GetReward));
      g_openGym->SetGetGameOverCb          (MakeCallback (&GetDone));
      g_openGym->SetGetExtraInfoCb         (MakeCallback (&GetInfo));
      g_openGym->SetExecuteActionsCb       (MakeCallback (&ExecuteAction));
      // First DrnnStep at 2.0 s — after TCP flows start and traces connected
      Simulator::Schedule (Seconds (2.0), &DrnnStep, 0.1);
    }

  // ═══════════════════════ run ═══════════════════════
  Simulator::Stop (Seconds (g_simTime));
  Simulator::Run ();

  g_tputFile.close ();
  g_cwndFile.close ();
  g_rttFile.close ();

  if (g_isDrnn && g_openGym)
    g_openGym->NotifySimulationEnd ();

  Simulator::Destroy ();

  // ═══════════════════════ summary ═══════════════════════
  std::cout << "\n=== WiFi Parking-Lot Simulation Complete ===\n"
            << "Transport:        " << transport        << "\n"
            << "Duration:         " << g_simTime        << " s\n"
            << "Flows:            " << NUM_FLOWS        << " TCP  +  1 UDP cross-traffic\n"
            << "Drops Link1:      " << g_dropsL1        << "  (R1→R2, 8 Mbps / 60p)\n"
            << "Drops Link2:      " << g_dropsL2        << "  (R2→Server, 4 Mbps / 40p)\n"
            << "WiFi MAC drops:   " << g_wifiMacDrops   << "  (MPDU max-retry failures)\n"
            << "WiFi PHY TX drops:" << g_wifiPhyTxDrops << "\n"
            << "WiFi PHY RX drops:" << g_wifiPhyRxDrops << "\n"
            << "CSV prefix:       " << "w_" + transport + "_" << "\n";

  return 0;
}

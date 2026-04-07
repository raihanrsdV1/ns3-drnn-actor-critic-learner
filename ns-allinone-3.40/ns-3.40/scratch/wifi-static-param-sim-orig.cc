/*
 * wifi-static-param-sim.cc -- Parameterizable WiFi 802.11n Static Simulation
 *
 * Topology (scalable WiFi infrastructure):
 *
 *   STA0 --+                                                 +-- Dst0
 *   STA1 --+-- (WiFi 802.11n) -- AP -- [10Mbps/10ms/64p] -- R1 -- [100Mbps/1ms] --+-- Dst1
 *   ...  --+                                                 +-- ...
 *
 *   nSTA = (nNodes-3)/2  WiFi client stations on left
 *   nDst = nNodes - 3 - nSTA  wired destination nodes on right
 *   (3 fixed nodes: AP, R1, and the topology itself)
 *   nFlows TCP flows assigned round-robin: flow i from STA(i%nSTA) to Dst(i%nDst)
 *
 * CLI Parameters:
 *   --nNodes        Total nodes incl. AP + R1          (20,40,60,80,100)
 *   --nFlows        Number of TCP flows                (10,20,30,40,50)
 *   --pps           Packets/sec per flow               (100,200,300,400,500)
 *   --coverageMult  Coverage area multiplier (1-5)     area = mult * 100m
 *   --speed         Node speed m/s (0 = static)        (0,5,10,15,20,25)
 *   --transport     cubic | reno | drnn
 *   --simTime       Simulation duration (s)            default 60
 *   --port          OpenGym port (drnn only)           default 5558
 *
 * Metrics (FlowMonitor ground-truth, printed as SUMMARY line):
 *   1. Network throughput (Mbps)
 *   2. End-to-end average delay (ms)
 *   3. Packet delivery ratio
 *   4. Packet drop ratio
 *   5. Energy consumption (J) — WiFi radio energy model
 *
 * DRNN OpenGym observation: 11 floats
 *   [cwnd0,rtt0,bif0, cwnd1,rtt1,bif1, cwnd2,rtt2,bif2, dropsQueue, wifiMacDrops]
 *   3 representative flows sampled from nFlows total.
 *   Action: 1 continuous float = target cwnd (bytes).
 *
 * CSV output (overwritten each run):
 *   ws_{transport}_throughput.csv -- Time,Total,Flow0..2,Drops_Q,Drops_WiFiMAC,Drops_WiFiPHY
 *   ws_{transport}_cwnd.csv      -- Time,Flow0..2_CWND
 *   ws_{transport}_rtt.csv       -- Time,Flow0..2_RTT
 *
 * Usage:
 *   ./ns3 run wifi-static-param-sim -- --transport=cubic --nNodes=40 --nFlows=20
 *   ./ns3 run wifi-static-param-sim -- --transport=drnn  --nNodes=40 --port=5558
 */

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/wifi-module.h"
#include "ns3/mobility-module.h"
#include "ns3/applications-module.h"
#include "ns3/traffic-control-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/energy-module.h"
#include "ns3/opengym-module.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("WifiStaticParamSim");

// ===================================================================
// CUSTOM CC: TcpDrnn  (continuous-control, same as wired-param-sim)
// ===================================================================
class TcpDrnn : public TcpCongestionOps
{
public:
  static TypeId GetTypeId (void);
  TcpDrnn ();
  TcpDrnn (const TcpDrnn &sock);
  ~TcpDrnn () override;

  std::string GetName () const override { return "TcpDrnn"; }

  uint32_t GetSsThresh (Ptr<const TcpSocketState> tcb,
                         uint32_t bytesInFlight) override;
  void IncreaseWindow (Ptr<TcpSocketState> tcb,
                       uint32_t segmentsAcked) override;
  Ptr<TcpCongestionOps> Fork () override;

  static float g_target_cwnd;
};

static const uint32_t DRNN_MIN_CWND =  2 * 1024;    //  2 KB
static const uint32_t DRNN_MAX_CWND = 80 * 1024;    // 80 KB

float TcpDrnn::g_target_cwnd = (float)(8 * 1024);   // 8 KB initial

NS_OBJECT_ENSURE_REGISTERED (TcpDrnn);

TypeId
TcpDrnn::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::TcpDrnn")
    .SetParent<TcpCongestionOps> ()
    .SetGroupName ("Internet")
    .AddConstructor<TcpDrnn> ();
  return tid;
}

TcpDrnn::TcpDrnn ()  : TcpCongestionOps () {}
TcpDrnn::TcpDrnn (const TcpDrnn &s) : TcpCongestionOps (s) {}
TcpDrnn::~TcpDrnn () {}

uint32_t
TcpDrnn::GetSsThresh (Ptr<const TcpSocketState> tcb,
                       uint32_t /* bytesInFlight */)
{
  return std::max<uint32_t> (DRNN_MIN_CWND, tcb->m_cWnd.Get () / 2);
}

void
TcpDrnn::IncreaseWindow (Ptr<TcpSocketState> tcb, uint32_t segmentsAcked)
{
  if (segmentsAcked == 0)
    return;

  uint32_t seg    = tcb->m_segmentSize;
  uint32_t target = static_cast<uint32_t> (
      std::max ((float)DRNN_MIN_CWND,
      std::min ((float)DRNN_MAX_CWND, g_target_cwnd)));
  target = std::max (2 * seg, (target / seg) * seg);
  tcb->m_cWnd = target;
}

Ptr<TcpCongestionOps>
TcpDrnn::Fork ()
{
  return CopyObject<TcpDrnn> (this);
}

// ======================= constants / tuning =======================
static const uint32_t NUM_REP = 3;   // representative flows for DRNN

// ======================= per-flow state =======================
struct FlowState
{
  uint64_t lastTotalRx  = 0;
  uint32_t cwnd         = 0;
  double   rttMs        = 0.0;
  uint32_t bytesInFlight = 0;
};

static FlowState g_rep[NUM_REP];
static uint32_t  g_repSinkIdx[NUM_REP];
static uint32_t  g_repNodeId[NUM_REP];

static std::vector<Ptr<PacketSink>> g_allSinks;

// ======================= drop counters =======================
static uint32_t g_dropsQueue       = 0;   // bottleneck queue drops
static uint32_t g_dropsQueueRev    = 0;   // reverse direction
static uint32_t g_wifiMacDrops     = 0;
static uint32_t g_wifiPhyTxDrops   = 0;
static uint32_t g_wifiPhyRxDrops   = 0;

// Per-interval deltas for CSV
static uint32_t g_lastDropsQueue      = 0;
static uint32_t g_lastDropsQueueRev   = 0;
static uint32_t g_lastWifiMacDrops    = 0;
static uint32_t g_lastWifiPhyTxDrops  = 0;
static uint32_t g_lastWifiPhyRxDrops  = 0;

// Per-step deltas for gym observation
static uint32_t g_gymLastDropsQueue   = 0;
static uint32_t g_gymLastWifiMacDrops = 0;

// ======================= energy sources =======================
static EnergySourceContainer g_energySources;

// ======================= CSV streams =======================
static std::ofstream g_tputFile;
static std::ofstream g_cwndFile;
static std::ofstream g_rttFile;

static double   g_simTime  = 60.0;
static bool     g_isDrnn   = false;

// ======================= OpenGym =======================
static Ptr<OpenGymInterface> g_openGym;
static uint32_t g_gymPort = 5558;
// Observation: 11 floats — per flow [cwnd, rtt, bif] × 3 + [dropsQueue, wifiMacDrops]
static const uint32_t OBS_DIM = NUM_REP * 3 + 2;

// ======================= trace callbacks =======================

static void
CwndTrace (uint32_t flowId, uint32_t /* oldVal */, uint32_t newVal)
{
  g_rep[flowId].cwnd = newVal;
}

static void
RttTrace (uint32_t flowId, Time /* oldVal */, Time newVal)
{
  g_rep[flowId].rttMs = newVal.GetMilliSeconds ();
}

static void
InFlightTrace (uint32_t flowId, uint32_t /* oldVal */, uint32_t newVal)
{
  g_rep[flowId].bytesInFlight = newVal;
}

static void
DropTraceQueue (Ptr<const QueueDiscItem> /* item */)
{
  ++g_dropsQueue;
}

static void
DropTraceQueueRev (Ptr<const QueueDiscItem> /* item */)
{
  ++g_dropsQueueRev;
}

// WiFi MAC drop: MPDU discarded after exhausting max retransmissions
static void
WifiMacDropTrace (WifiMacDropReason /* reason */,
                  Ptr<const WifiMpdu> /* mpdu */)
{
  ++g_wifiMacDrops;
}

// WiFi PHY TX drop
static void
WifiPhyTxDropTrace (Ptr<const Packet> /* p */)
{
  ++g_wifiPhyTxDrops;
}

// WiFi PHY RX drop
static void
WifiPhyRxDropTrace (Ptr<const Packet> /* p */,
                    WifiPhyRxfailureReason /* reason */)
{
  ++g_wifiPhyRxDrops;
}

// Connect WiFi drop traces on all STAs and the AP
static void
ConnectWifiDropTraces (NodeContainer wifiNodes)
{
  for (uint32_t i = 0; i < wifiNodes.GetN (); ++i)
    {
      Ptr<NetDevice> dev = wifiNodes.Get (i)->GetDevice (0);
      Ptr<WifiNetDevice> wifiDev = DynamicCast<WifiNetDevice> (dev);
      if (!wifiDev)
        continue;

      // MAC drop trace
      Ptr<WifiMac> mac = wifiDev->GetMac ();
      mac->TraceConnectWithoutContext ("DroppedMpdu",
                                       MakeCallback (&WifiMacDropTrace));

      // PHY drop traces
      Ptr<WifiPhy> phy = wifiDev->GetPhy ();
      phy->TraceConnectWithoutContext ("PhyTxDrop",
                                       MakeCallback (&WifiPhyTxDropTrace));
      phy->TraceConnectWithoutContext ("PhyRxDrop",
                                       MakeCallback (&WifiPhyRxDropTrace));
    }
  NS_LOG_UNCOND ("WiFi drop traces connected for "
                 << wifiNodes.GetN () << " nodes");
}

// ======================= OpenGym callbacks =======================

Ptr<OpenGymSpace>
GetObsSpace ()
{
  std::vector<uint32_t> shape = {(uint32_t)OBS_DIM};
  Ptr<OpenGymBoxSpace> space = CreateObject<OpenGymBoxSpace> (
      0.0f, 1e9f, shape, TypeNameGet<float> ());
  return space;
}

Ptr<OpenGymSpace>
GetActSpace ()
{
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
  for (uint32_t i = 0; i < NUM_REP; ++i)
    {
      box->AddValue ((float)g_rep[i].cwnd);
      box->AddValue ((float)g_rep[i].rttMs);
      box->AddValue ((float)g_rep[i].bytesInFlight);
    }
  // Drop deltas
  uint32_t dQ    = g_dropsQueue  - g_gymLastDropsQueue;
  uint32_t dWifi = g_wifiMacDrops - g_gymLastWifiMacDrops;
  g_gymLastDropsQueue   = g_dropsQueue;
  g_gymLastWifiMacDrops = g_wifiMacDrops;
  box->AddValue ((float)dQ);
  box->AddValue ((float)dWifi);
  return box;
}

float
GetReward ()
{
  // C++ reward is a hint — Python agent computes its own
  double totalTput = 0.0;
  int    active    = 0;
  double avgRtt    = 0.0;
  for (uint32_t i = 0; i < NUM_REP; ++i)
    {
      if (g_rep[i].cwnd > 0 && g_rep[i].rttMs > 0)
        {
          double rtt_s = g_rep[i].rttMs / 1000.0;
          totalTput += (double)g_rep[i].cwnd / rtt_s * 8.0 / 1e6;
          avgRtt    += g_rep[i].rttMs;
          ++active;
        }
    }
  if (active == 0)
    return 0.0f;
  avgRtt /= active;

  double util   = std::min (totalTput / 10.0, 1.0);   // 10 Mbps bottleneck
  double latPen = 0.8 * std::max (0.0, avgRtt / 30.0 - 1.0);
  return (float)(util - latPen);
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
  Ptr<OpenGymBoxContainer<float>> box =
    DynamicCast<OpenGymBoxContainer<float>> (action);
  if (box && box->GetData ().size () >= 1)
    {
      float val = box->GetData ()[0];
      TcpDrnn::g_target_cwnd =
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

// ======================= periodic measurement =======================

static void
CalcThroughput ()
{
  double now = Simulator::Now ().GetSeconds ();
  if (now >= g_simTime)
    return;

  // Per-representative-flow throughput
  double flowTput[NUM_REP];
  double total = 0.0;
  for (uint32_t i = 0; i < NUM_REP; ++i)
    {
      uint32_t sIdx = g_repSinkIdx[i];
      if (sIdx < g_allSinks.size () && g_allSinks[sIdx])
        {
          uint64_t rx = g_allSinks[sIdx]->GetTotalRx ();
          double mbps = (rx - g_rep[i].lastTotalRx) * 8.0 / 0.1 / 1e6;
          g_rep[i].lastTotalRx = rx;
          flowTput[i] = mbps;
        }
      else
        flowTput[i] = 0.0;
      total += flowTput[i];
    }

  // Drop deltas for CSV
  uint32_t dQ       = g_dropsQueue     - g_lastDropsQueue;
  g_lastDropsQueue  = g_dropsQueue;
  uint32_t dQRev    = g_dropsQueueRev  - g_lastDropsQueueRev;
  g_lastDropsQueueRev = g_dropsQueueRev;
  uint32_t dWifiMac = g_wifiMacDrops   - g_lastWifiMacDrops;
  g_lastWifiMacDrops  = g_wifiMacDrops;
  uint32_t dWifiPhy = (g_wifiPhyTxDrops - g_lastWifiPhyTxDrops)
                    + (g_wifiPhyRxDrops  - g_lastWifiPhyRxDrops);
  g_lastWifiPhyTxDrops = g_wifiPhyTxDrops;
  g_lastWifiPhyRxDrops = g_wifiPhyRxDrops;

  g_tputFile << now << "," << total;
  for (uint32_t i = 0; i < NUM_REP; ++i)
    g_tputFile << "," << flowTput[i];
  g_tputFile << "," << (dQ + dQRev) << "," << dWifiMac << "," << dWifiPhy << "\n";

  g_cwndFile << now;
  for (uint32_t i = 0; i < NUM_REP; ++i)
    g_cwndFile << "," << g_rep[i].cwnd;
  g_cwndFile << "\n";

  g_rttFile << now;
  for (uint32_t i = 0; i < NUM_REP; ++i)
    g_rttFile << "," << g_rep[i].rttMs;
  g_rttFile << "\n";

  Simulator::Schedule (Seconds (0.1), &CalcThroughput);
}

static void
StartMeasurement ()
{
  for (uint32_t i = 0; i < NUM_REP; ++i)
    {
      uint32_t sIdx = g_repSinkIdx[i];
      if (sIdx < g_allSinks.size () && g_allSinks[sIdx])
        g_rep[i].lastTotalRx = g_allSinks[sIdx]->GetTotalRx ();
    }
  Simulator::Schedule (Seconds (0.1), &CalcThroughput);
}

// ======================= connect per-socket traces =======================

static void
ConnectSocketTraces ()
{
  for (uint32_t i = 0; i < NUM_REP; ++i)
    {
      std::ostringstream cwndPath, rttPath, ifPath;
      cwndPath << "/NodeList/" << g_repNodeId[i]
               << "/$ns3::TcpL4Protocol/SocketList/0/CongestionWindow";
      rttPath  << "/NodeList/" << g_repNodeId[i]
               << "/$ns3::TcpL4Protocol/SocketList/0/RTT";
      ifPath   << "/NodeList/" << g_repNodeId[i]
               << "/$ns3::TcpL4Protocol/SocketList/0/BytesInFlight";

      Config::ConnectWithoutContext (cwndPath.str (),
                                    MakeBoundCallback (&CwndTrace, i));
      Config::ConnectWithoutContext (rttPath.str (),
                                    MakeBoundCallback (&RttTrace, i));
      Config::ConnectWithoutContext (ifPath.str (),
                                    MakeBoundCallback (&InFlightTrace, i));
    }
  NS_LOG_UNCOND ("Socket traces connected for " << NUM_REP << " representative flows");
}

// ===================================================================
//  main
// ===================================================================

int
main (int argc, char *argv[])
{
  // -- CLI parameters --
  std::string transport = "cubic";
  uint32_t nNodes       = 40;
  uint32_t nFlows       = 20;
  uint32_t pps          = 200;
  uint32_t coverageMult = 1;
  double   speed        = 0.0;

  CommandLine cmd (__FILE__);
  cmd.AddValue ("transport",    "TCP variant: cubic | reno | drnn",     transport);
  cmd.AddValue ("nNodes",       "Total nodes (incl AP, R1)",            nNodes);
  cmd.AddValue ("nFlows",       "Number of TCP flows",                  nFlows);
  cmd.AddValue ("pps",          "Packets per second per flow",          pps);
  cmd.AddValue ("coverageMult", "Coverage area multiplier (1-5)",       coverageMult);
  cmd.AddValue ("speed",        "Node speed m/s (0=static)",            speed);
  cmd.AddValue ("simTime",      "Simulation duration (s)",              g_simTime);
  cmd.AddValue ("port",         "OpenGym port (drnn only)",             g_gymPort);
  cmd.Parse (argc, argv);

  g_isDrnn = (transport == "drnn");

  // Derive STA count and destination count
  // nNodes = nSTA + 1 (AP) + 1 (R1) + nDst
  // Split evenly: nSTA ~ half, nDst ~ half
  uint32_t nSTA = (nNodes - 2) / 2;
  uint32_t nDst = nNodes - 2 - nSTA;

  NS_ABORT_MSG_IF (nSTA < 1, "Need at least 1 STA (nNodes >= 4)");
  NS_ABORT_MSG_IF (nDst < 1, "Need at least 1 Dst (nNodes >= 4)");
  NS_ABORT_MSG_IF (nFlows < 1, "Need at least 1 flow");

  if (nFlows > nSTA)
    NS_LOG_UNCOND ("NOTE: nFlows=" << nFlows << " > nSTA=" << nSTA
                   << "; flows assigned round-robin.");

  // -- TCP variant --
  if (transport == "cubic")
    Config::SetDefault ("ns3::TcpL4Protocol::SocketType",
                        TypeIdValue (TcpCubic::GetTypeId ()));
  else if (transport == "reno")
    Config::SetDefault ("ns3::TcpL4Protocol::SocketType",
                        TypeIdValue (TcpNewReno::GetTypeId ()));
  else if (transport == "drnn")
    Config::SetDefault ("ns3::TcpL4Protocol::SocketType",
                        TypeIdValue (TcpDrnn::GetTypeId ()));
  else
    NS_FATAL_ERROR ("Unknown transport: " << transport);

  // -- TCP knobs --
  Config::SetDefault ("ns3::TcpSocket::SegmentSize",  UintegerValue (1024));
  Config::SetDefault ("ns3::TcpSocket::SndBufSize",   UintegerValue (1 << 20));
  Config::SetDefault ("ns3::TcpSocket::RcvBufSize",   UintegerValue (1 << 20));
  Config::SetDefault ("ns3::TcpSocket::InitialSlowStartThreshold",
                      UintegerValue (10 * 1024 * 1024));
  Config::SetDefault ("ns3::DropTailQueue<Packet>::MaxSize",
                      QueueSizeValue (QueueSize ("1p")));

  // ======================= nodes =======================
  NodeContainer staNodes;  staNodes.Create (nSTA);
  NodeContainer apNode;    apNode.Create (1);
  NodeContainer r1Node;    r1Node.Create (1);
  NodeContainer dstNodes;  dstNodes.Create (nDst);

  // Pick 3 representative flows
  uint32_t repIdx[NUM_REP] = {0, nFlows / 3, 2 * nFlows / 3};
  for (uint32_t r = 0; r < NUM_REP; ++r)
    {
      g_repSinkIdx[r] = repIdx[r];
      g_repNodeId[r]  = staNodes.Get (repIdx[r] % nSTA)->GetId ();
    }

  // ======================= WiFi 802.11n =======================
  // Channel with log-distance propagation loss (realistic fading)
  YansWifiChannelHelper wifiChannel;
  wifiChannel.SetPropagationDelay ("ns3::ConstantSpeedPropagationDelayModel");
  // Log-distance: PL(d) = PL(d0) + 10*n*log10(d/d0), n=3.0 (indoor/urban)
  wifiChannel.AddPropagationLoss ("ns3::LogDistancePropagationLossModel",
                                  "Exponent", DoubleValue (3.0),
                                  "ReferenceDistance", DoubleValue (1.0),
                                  "ReferenceLoss", DoubleValue (46.67));

  YansWifiPhyHelper wifiPhy;
  wifiPhy.SetChannel (wifiChannel.Create ());
  // Default Tx power ~20 dBm; Tx range ~ 100m with this loss model

  WifiHelper wifi;
  wifi.SetStandard (WIFI_STANDARD_80211n);
  // MinstrelHt is accurate but expensive for large sweeps (many stations).
  // IdealWifiManager is much lighter and avoids very long runtimes / SIGKILL
  // in dense scenarios (e.g., 60+ nodes) while preserving comparative trends.
  wifi.SetRemoteStationManager ("ns3::IdealWifiManager");

  // Cap retransmission attempts to prevent event explosion under heavy contention.
  Config::SetDefault ("ns3::WifiRemoteStationManager::MaxSlrc", UintegerValue (4));
  Config::SetDefault ("ns3::WifiRemoteStationManager::MaxSsrc", UintegerValue (4));

  WifiMacHelper mac;
  Ssid ssid = Ssid ("wifi-static-sim");

  // STA devices
  mac.SetType ("ns3::StaWifiMac",
               "Ssid", SsidValue (ssid),
               "ActiveProbing", BooleanValue (false));
  NetDeviceContainer staDevices = wifi.Install (wifiPhy, mac, staNodes);

  // AP device
  mac.SetType ("ns3::ApWifiMac", "Ssid", SsidValue (ssid));
  NetDeviceContainer apDevice = wifi.Install (wifiPhy, mac, apNode);

  // ======================= mobility =======================
  // Coverage area: coverageMult * 100m square
  double areaSize = coverageMult * 100.0;

  // Grid placement for STAs within coverage area
  MobilityHelper staMobility;
  // Compute grid: ceil(sqrt(nSTA)) x ceil(sqrt(nSTA))
  uint32_t gridSide = (uint32_t)std::ceil (std::sqrt ((double)nSTA));
  double gridSpacing = (gridSide > 1) ? areaSize / (gridSide - 1) : 0.0;
  if (gridSide <= 1)
    gridSpacing = 0.0;

  Ptr<ListPositionAllocator> staPos = CreateObject<ListPositionAllocator> ();
  for (uint32_t i = 0; i < nSTA; ++i)
    {
      double x = (i % gridSide) * gridSpacing;
      double y = (i / gridSide) * gridSpacing;
      staPos->Add (Vector (x, y, 1.5));
    }
  staMobility.SetPositionAllocator (staPos);

  if (speed > 0.0)
    {
      // RandomWalk2d within coverage bounds
      std::ostringstream boundsStr;
      boundsStr << "0|" << areaSize << "|0|" << areaSize;
      std::ostringstream speedStr;
      speedStr << "ns3::ConstantRandomVariable[Constant=" << speed << "]";
      staMobility.SetMobilityModel ("ns3::RandomWalk2dMobilityModel",
                                    "Bounds", RectangleValue (
                                        Rectangle (0, areaSize, 0, areaSize)),
                                    "Speed", StringValue (speedStr.str ()),
                                    "Distance", DoubleValue (20.0));
    }
  else
    {
      staMobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
    }
  staMobility.Install (staNodes);

  // AP at center of coverage area
  MobilityHelper apMobility;
  Ptr<ListPositionAllocator> apPos = CreateObject<ListPositionAllocator> ();
  apPos->Add (Vector (areaSize / 2.0, areaSize / 2.0, 3.0));
  apMobility.SetPositionAllocator (apPos);
  apMobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  apMobility.Install (apNode);

  // ======================= wired links =======================

  // AP <-> R1 : bottleneck (10 Mbps, 10 ms)
  PointToPointHelper p2pBn;
  p2pBn.SetDeviceAttribute  ("DataRate", StringValue ("10Mbps"));
  p2pBn.SetChannelAttribute ("Delay",    StringValue ("10ms"));
  NetDeviceContainer bnDev = p2pBn.Install (apNode.Get (0), r1Node.Get (0));

  // R1 <-> Dst_j : fast access links (100 Mbps, 1 ms)
  PointToPointHelper p2pAccess;
  p2pAccess.SetDeviceAttribute  ("DataRate", StringValue ("100Mbps"));
  p2pAccess.SetChannelAttribute ("Delay",    StringValue ("1ms"));

  std::vector<NetDeviceContainer> dstDevs (nDst);
  for (uint32_t j = 0; j < nDst; ++j)
    dstDevs[j] = p2pAccess.Install (r1Node.Get (0), dstNodes.Get (j));

  // ======================= internet stack =======================
  InternetStackHelper internet;
  internet.Install (staNodes);
  internet.Install (apNode);
  internet.Install (r1Node);
  internet.Install (dstNodes);

  // ======================= traffic control =======================
  // Install BEFORE Ipv4AddressHelper::Assign()

  // Forward direction AP->R1 (64-pkt queue — main bottleneck)
  TrafficControlHelper tchFwd;
  tchFwd.SetRootQueueDisc ("ns3::FifoQueueDisc",
                            "MaxSize", StringValue ("64p"));
  QueueDiscContainer qdFwd = tchFwd.Install (bnDev.Get (0));
  qdFwd.Get (0)->TraceConnectWithoutContext ("Drop",
                                             MakeCallback (&DropTraceQueue));

  // Reverse direction R1->AP (ACK packets, 64-pkt queue)
  TrafficControlHelper tchRev;
  tchRev.SetRootQueueDisc ("ns3::FifoQueueDisc",
                            "MaxSize", StringValue ("64p"));
  QueueDiscContainer qdRev = tchRev.Install (bnDev.Get (1));
  qdRev.Get (0)->TraceConnectWithoutContext ("Drop",
                                             MakeCallback (&DropTraceQueueRev));

  // ======================= IP addresses =======================
  Ipv4AddressHelper ipv4;

  // WiFi subnet 10.1.1.0/24
  ipv4.SetBase ("10.1.1.0", "255.255.255.0");
  ipv4.Assign (staDevices);
  ipv4.Assign (apDevice);

  // AP <-> R1  10.1.2.0/24
  ipv4.SetBase ("10.1.2.0", "255.255.255.0");
  ipv4.Assign (bnDev);

  // R1 <-> Dst_j  10.2.(10+j).0/24
  std::vector<Ipv4Address> dstAddr (nDst);
  for (uint32_t j = 0; j < nDst; ++j)
    {
      std::ostringstream base;
      base << "10.2." << (j + 10) << ".0";
      ipv4.SetBase (base.str ().c_str (), "255.255.255.0");
      Ipv4InterfaceContainer iface = ipv4.Assign (dstDevs[j]);
      dstAddr[j] = iface.GetAddress (1);   // destination node's IP
    }

  Ipv4GlobalRoutingHelper::PopulateRoutingTables ();

  // ======================= energy model =======================
  // Install BasicEnergySource + WifiRadioEnergyModel on all WiFi nodes
  BasicEnergySourceHelper energyHelper;
  energyHelper.Set ("BasicEnergySourceInitialEnergyJ", DoubleValue (1000.0));

  // STAs
  EnergySourceContainer staEnergy = energyHelper.Install (staNodes);
  // AP
  EnergySourceContainer apEnergy = energyHelper.Install (apNode);

  WifiRadioEnergyModelHelper radioEnergyHelper;
  // Default power consumption values (idle=0.273W, tx=1.14W, rx=0.582W, sleep=0.0033W)

  for (uint32_t i = 0; i < staNodes.GetN (); ++i)
    {
      DeviceEnergyModelContainer devModels =
        radioEnergyHelper.Install (staDevices.Get (i), staEnergy.Get (i));
    }
  DeviceEnergyModelContainer apDevModel =
    radioEnergyHelper.Install (apDevice.Get (0), apEnergy.Get (0));

  // Store all energy sources for final accounting
  g_energySources.Add (staEnergy);
  g_energySources.Add (apEnergy);

  // ======================= TCP applications =======================
  uint16_t basePort = 9000;

  // Data rate per flow: pps * 1024-byte packets
  uint64_t dataRateBps = (uint64_t)pps * 1024 * 8;
  std::ostringstream drStr;
  drStr << dataRateBps << "bps";

  for (uint32_t i = 0; i < nFlows; ++i)
    {
      uint16_t port   = basePort + i;
      uint32_t staIdx = i % nSTA;
      uint32_t dstIdx = i % nDst;

      // PacketSink on destination
      PacketSinkHelper sinkHelper ("ns3::TcpSocketFactory",
          InetSocketAddress (Ipv4Address::GetAny (), port));
      ApplicationContainer sinkApp = sinkHelper.Install (dstNodes.Get (dstIdx));
      sinkApp.Start (Seconds (0.0));
      sinkApp.Stop  (Seconds (g_simTime));
      g_allSinks.push_back (DynamicCast<PacketSink> (sinkApp.Get (0)));

      // OnOff TCP source from STA (always-on, rate = pps * 1024 bytes/s)
      OnOffHelper onoff ("ns3::TcpSocketFactory",
          InetSocketAddress (dstAddr[dstIdx], port));
      onoff.SetAttribute ("DataRate",   StringValue (drStr.str ()));
      onoff.SetAttribute ("PacketSize", UintegerValue (1024));
      onoff.SetAttribute ("OnTime",  StringValue (
          "ns3::ConstantRandomVariable[Constant=1000000]"));
      onoff.SetAttribute ("OffTime", StringValue (
          "ns3::ConstantRandomVariable[Constant=0]"));

      ApplicationContainer srcApp = onoff.Install (staNodes.Get (staIdx));
      srcApp.Start (Seconds (0.5 + i * 0.01));
      srcApp.Stop  (Seconds (g_simTime));
    }

  // ======================= FlowMonitor =======================
  FlowMonitorHelper flowMonHelper;
  Ptr<FlowMonitor> flowMon = flowMonHelper.InstallAll ();

  // ======================= CSV files =======================
  std::string pfx = "ws_" + transport + "_";

  g_tputFile.open (pfx + "throughput.csv");
  g_tputFile << "Time,Total";
  for (uint32_t r = 0; r < NUM_REP; ++r) g_tputFile << ",Flow" << r;
  g_tputFile << ",Drops_Q,Drops_WiFiMAC,Drops_WiFiPHY\n";

  g_cwndFile.open (pfx + "cwnd.csv");
  g_cwndFile << "Time";
  for (uint32_t r = 0; r < NUM_REP; ++r) g_cwndFile << ",Flow" << r << "_CWND";
  g_cwndFile << "\n";

  g_rttFile.open (pfx + "rtt.csv");
  g_rttFile << "Time";
  for (uint32_t r = 0; r < NUM_REP; ++r) g_rttFile << ",Flow" << r << "_RTT";
  g_rttFile << "\n";

  // ======================= schedule traces =======================
  // Connect WiFi drop traces immediately (WiFi devices already installed)
  NodeContainer allWifi;
  allWifi.Add (staNodes);
  allWifi.Add (apNode);
  Simulator::Schedule (Seconds (0.1), &ConnectWifiDropTraces, allWifi);

  Simulator::Schedule (Seconds (1.5), &ConnectSocketTraces);
  Simulator::Schedule (Seconds (1.6), &StartMeasurement);

  // ======================= OpenGym (drnn only) =======================
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
      Simulator::Schedule (Seconds (2.0), &DrnnStep, 0.1);
    }

  // ======================= run =======================
  Simulator::Stop (Seconds (g_simTime));
  Simulator::Run ();

  g_tputFile.close ();
  g_cwndFile.close ();
  g_rttFile.close ();

  if (g_isDrnn && g_openGym)
    g_openGym->NotifySimulationEnd ();

  // ======================= Energy accounting =======================
  double totalEnergyJ = 0.0;
  for (uint32_t i = 0; i < g_energySources.GetN (); ++i)
    {
      Ptr<BasicEnergySource> src =
        DynamicCast<BasicEnergySource> (g_energySources.Get (i));
      if (src)
        {
          double initial   = src->GetInitialEnergy ();
          double remaining = src->GetRemainingEnergy ();
          totalEnergyJ += (initial - remaining);
        }
    }

  // ======================= FlowMonitor summary =======================
  flowMon->CheckForLostPackets ();
  Ptr<Ipv4FlowClassifier> classifier =
    DynamicCast<Ipv4FlowClassifier> (flowMonHelper.GetClassifier ());

  uint64_t totalTxPkts = 0, totalRxPkts = 0, totalLostPkts = 0;
  double   totalRxBytes = 0.0, totalDelayMs = 0.0;
  uint64_t rxPktsForDelay = 0;

  auto stats = flowMon->GetFlowStats ();
  for (auto it = stats.begin (); it != stats.end (); ++it)
    {
      Ipv4FlowClassifier::FiveTuple t = classifier->FindFlow (it->first);
      if (t.destinationPort >= basePort
          && t.destinationPort < basePort + nFlows)
        {
          totalTxPkts   += it->second.txPackets;
          totalRxPkts   += it->second.rxPackets;
          totalLostPkts += it->second.lostPackets;
          totalRxBytes  += it->second.rxBytes;
          if (it->second.rxPackets > 0)
            {
              totalDelayMs   += it->second.delaySum.GetSeconds () * 1000.0;
              rxPktsForDelay += it->second.rxPackets;
            }
        }
    }

  double throughputMbps = totalRxBytes * 8.0 / g_simTime / 1e6;
  double avgDelayMs = (rxPktsForDelay > 0)
                      ? totalDelayMs / rxPktsForDelay : 0.0;
  double pdr       = (totalTxPkts > 0)
                      ? (double)totalRxPkts / totalTxPkts : 0.0;
  double dropRatio = 1.0 - pdr;

  uint32_t totalDrops = g_dropsQueue + g_dropsQueueRev
                      + g_wifiMacDrops + g_wifiPhyTxDrops + g_wifiPhyRxDrops;

  Simulator::Destroy ();

  // Human-readable
  std::cout << "\n=== WiFi 802.11n Static Simulation Complete ===\n"
    << "Transport:     " << transport << "\n"
    << "Nodes:         " << nNodes << " (" << nSTA << " STAs + 1 AP + 1 R1 + "
                         << nDst << " Dst)\n"
    << "Flows:         " << nFlows << "\n"
    << "PPS/flow:      " << pps << "\n"
    << "Coverage:      " << coverageMult << "x (" << areaSize << " m)\n"
    << "Speed:         " << speed << " m/s\n"
    << "Duration:      " << g_simTime << " s\n"
    << "Throughput:    " << throughputMbps << " Mbps\n"
    << "Avg Delay:     " << avgDelayMs << " ms\n"
    << "PDR:           " << pdr << "\n"
    << "Drop Ratio:    " << dropRatio << "\n"
    << "Queue Drops:   " << g_dropsQueue << " (fwd) + " << g_dropsQueueRev << " (rev)\n"
    << "WiFi MAC Drop: " << g_wifiMacDrops << "\n"
    << "WiFi PHY Drop: " << g_wifiPhyTxDrops << " (tx) + " << g_wifiPhyRxDrops << " (rx)\n"
    << "FlowMon Lost:  " << totalLostPkts << "\n"
    << "Energy:        " << totalEnergyJ << " J\n"
    << "CSV prefix:    " << pfx << "\n";

  // Machine-readable (shell script greps for this)
  // SUMMARY,transport,nNodes,nFlows,pps,coverageMult,speed,
  //   throughputMbps,avgDelayMs,pdr,dropRatio,totalDrops,txPkts,rxPkts,energyJ
  std::cout << "SUMMARY,"
    << transport      << ","
    << nNodes         << ","
    << nFlows         << ","
    << pps            << ","
    << coverageMult   << ","
    << speed          << ","
    << throughputMbps << ","
    << avgDelayMs     << ","
    << pdr            << ","
    << dropRatio      << ","
    << totalDrops     << ","
    << totalTxPkts    << ","
    << totalRxPkts    << ","
    << totalEnergyJ   << "\n";

  return 0;
}

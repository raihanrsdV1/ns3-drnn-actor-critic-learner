/*
 * wired-param-sim.cc -- Parameterizable Wired Dumbbell TCP Simulation
 *
 * Topology (scalable dumbbell):
 *
 *   Src0 --+                                          +-- Dst0
 *   Src1 --+-- [100Mbps/2ms] -- R1 --[10Mbps/20ms/64p]-- R2 -- [100Mbps/2ms] --+-- Dst1
 *   ...  --+                                          +-- ...
 *
 *   nSrc = (nNodes-2)/2  source nodes on left
 *   nDst = nNodes-2-nSrc  destination nodes on right
 *   nFlows TCP flows assigned round-robin: flow i from Src(i%nSrc) to Dst(i%nDst)
 *
 * CLI Parameters:
 *   --nNodes     Total nodes incl. 2 routers  (20,40,60,80,100)
 *   --nFlows     Number of TCP flows          (10,20,30,40,50)
 *   --pps        Packets/sec per flow         (100,200,300,400,500)
 *   --transport  cubic | reno | drnn
 *   --simTime    Simulation duration (s)      default 60
 *   --port       OpenGym port (drnn only)     default 5557
 *
 * Metrics (FlowMonitor ground-truth, printed as SUMMARY line):
 *   1. Network throughput (Mbps)
 *   2. End-to-end average delay (ms)
 *   3. Packet delivery ratio
 *   4. Packet drop ratio
 *
 * DRNN OpenGym observation: 11 floats
 *   [cwnd0,rtt0,bif0, cwnd1,rtt1,bif1, cwnd2,rtt2,bif2, dropsL1, dropsL2]
 *   3 representative flows sampled from nFlows total.
 *   Action: 1 continuous float = target cwnd (bytes).
 *
 * CSV output (overwritten each run):
 *   wd_{transport}_throughput.csv -- Time,Total,Flow0..2,Drops_L1,Drops_L2
 *   wd_{transport}_cwnd.csv      -- Time,Flow0..2_CWND
 *   wd_{transport}_rtt.csv       -- Time,Flow0..2_RTT
 *
 * Usage:
 *   ./ns3 run wired-param-sim -- --transport=cubic --nNodes=40 --nFlows=20 --pps=200
 */

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/applications-module.h"
#include "ns3/traffic-control-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/opengym-module.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("WiredParamSim");

// ===================================================================
// CUSTOM CC: TcpDrnn  (continuous-control, same design as wifi-sim)
//
// Python SAC agent outputs one float: g_target_cwnd in bytes.
// IncreaseWindow() clamps and assigns it directly to tcb->m_cWnd.
// On loss: standard cwnd/2 safety halving.
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

static const uint32_t DRNN_MIN_CWND =  2 * 1024;    //  2 KB  (2 segments)
static const uint32_t DRNN_MAX_CWND = 100 * 1024;   // 100 KB

float TcpDrnn::g_target_cwnd = (float)(10 * 1024);  // 10 KB initial

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

// ======================= constants & globals =======================
static const uint32_t NUM_REP = 3;   // representative flows for agent

struct FlowState
{
  uint64_t lastTotalRx   = 0;
  uint32_t cwnd          = 0;
  double   rttMs         = 0.0;
  uint32_t bytesInFlight = 0;
};

static FlowState  g_rep[NUM_REP];
static uint32_t   g_repNodeId[NUM_REP];
static uint32_t   g_repSinkIdx[NUM_REP];

// All sinks (for aggregate throughput)
static std::vector<Ptr<PacketSink>> g_allSinks;
static std::vector<uint64_t>        g_lastRx;

// Drop counters
static uint32_t g_dropsL1      = 0;   // R1->R2 (forward data)
static uint32_t g_dropsL2      = 0;   // R2->R1 (reverse ACKs)
static uint32_t g_lastDropsL1  = 0;
static uint32_t g_lastDropsL2  = 0;
static uint32_t g_gymLastDropsL1 = 0;
static uint32_t g_gymLastDropsL2 = 0;

// CSV streams
static std::ofstream g_tputFile;
static std::ofstream g_cwndFile;
static std::ofstream g_rttFile;

// Config
static double   g_simTime = 60.0;
static bool     g_isDrnn  = false;
static uint32_t g_nFlows  = 20;

// OpenGym
static Ptr<OpenGymInterface> g_openGym;
static uint32_t g_gymPort = 5557;
static const uint32_t OBS_DIM = NUM_REP * 3 + 2;  // 11

// ======================= trace callbacks =======================

static void
CwndTrace (uint32_t idx, uint32_t /* old */, uint32_t newVal)
{
  g_rep[idx].cwnd = newVal;
}

static void
RttTrace (uint32_t idx, Time /* old */, Time newVal)
{
  g_rep[idx].rttMs = newVal.GetMilliSeconds ();
}

static void
InFlightTrace (uint32_t idx, uint32_t /* old */, uint32_t newVal)
{
  g_rep[idx].bytesInFlight = newVal;
}

static void DropTraceL1 (Ptr<const QueueDiscItem>) { ++g_dropsL1; }
static void DropTraceL2 (Ptr<const QueueDiscItem>) { ++g_dropsL2; }

// ======================= OpenGym callbacks =======================

Ptr<OpenGymSpace>
GetObsSpace ()
{
  std::vector<uint32_t> shape = {(uint32_t)OBS_DIM};
  return CreateObject<OpenGymBoxSpace> (0.0f, 1e9f, shape,
                                        TypeNameGet<float> ());
}

Ptr<OpenGymSpace>
GetActSpace ()
{
  std::vector<uint32_t> shape = {1};
  return CreateObject<OpenGymBoxSpace> (
      (float)DRNN_MIN_CWND, (float)DRNN_MAX_CWND,
      shape, TypeNameGet<float> ());
}

Ptr<OpenGymDataContainer>
GetObs ()
{
  Ptr<OpenGymBoxContainer<float>> box =
    CreateObject<OpenGymBoxContainer<float>> ();
  for (uint32_t r = 0; r < NUM_REP; ++r)
    {
      box->AddValue ((float)g_rep[r].cwnd);
      box->AddValue ((float)g_rep[r].rttMs);
      box->AddValue ((float)g_rep[r].bytesInFlight);
    }
  uint32_t dL1 = g_dropsL1 - g_gymLastDropsL1;
  uint32_t dL2 = g_dropsL2 - g_gymLastDropsL2;
  g_gymLastDropsL1 = g_dropsL1;
  g_gymLastDropsL2 = g_dropsL2;
  box->AddValue ((float)dL1);
  box->AddValue ((float)dL2);
  return box;
}

float
GetReward ()
{
  // Simple utilization reward — Python agent overrides with its own.
  double totalTput = 0.0;
  int active = 0;
  for (uint32_t r = 0; r < NUM_REP; ++r)
    {
      if (g_rep[r].cwnd > 0 && g_rep[r].rttMs > 0)
        {
          totalTput += (double)g_rep[r].cwnd
                       / (g_rep[r].rttMs / 1000.0) * 8.0 / 1e6;
          ++active;
        }
    }
  if (active == 0) return 0.0f;
  return (float)std::min (totalTput / 10.0, 1.0);
}

bool GetDone ()    { return Simulator::Now ().GetSeconds () >= g_simTime; }
std::string GetInfo () { return ""; }

bool
ExecuteAction (Ptr<OpenGymDataContainer> action)
{
  Ptr<OpenGymBoxContainer<float>> box =
    DynamicCast<OpenGymBoxContainer<float>> (action);
  if (box && box->GetData ().size () >= 1)
    {
      TcpDrnn::g_target_cwnd =
        std::max ((float)DRNN_MIN_CWND,
        std::min ((float)DRNN_MAX_CWND, box->GetData ()[0]));
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
  if (now >= g_simTime) return;

  // Aggregate throughput across ALL sinks
  double total = 0.0;
  for (size_t i = 0; i < g_allSinks.size (); ++i)
    {
      uint64_t rx = g_allSinks[i]->GetTotalRx ();
      total += (rx - g_lastRx[i]) * 8.0 / 0.1 / 1e6;
      g_lastRx[i] = rx;
    }

  // Per representative-flow throughput
  double repTput[NUM_REP];
  for (uint32_t r = 0; r < NUM_REP; ++r)
    {
      uint64_t rx = g_allSinks[g_repSinkIdx[r]]->GetTotalRx ();
      repTput[r] = (rx - g_rep[r].lastTotalRx) * 8.0 / 0.1 / 1e6;
      g_rep[r].lastTotalRx = rx;
    }

  uint32_t dL1 = g_dropsL1 - g_lastDropsL1;  g_lastDropsL1 = g_dropsL1;
  uint32_t dL2 = g_dropsL2 - g_lastDropsL2;  g_lastDropsL2 = g_dropsL2;

  g_tputFile << now << "," << total;
  for (uint32_t r = 0; r < NUM_REP; ++r) g_tputFile << "," << repTput[r];
  g_tputFile << "," << dL1 << "," << dL2 << "\n";

  g_cwndFile << now;
  for (uint32_t r = 0; r < NUM_REP; ++r) g_cwndFile << "," << g_rep[r].cwnd;
  g_cwndFile << "\n";

  g_rttFile << now;
  for (uint32_t r = 0; r < NUM_REP; ++r) g_rttFile << "," << g_rep[r].rttMs;
  g_rttFile << "\n";

  Simulator::Schedule (Seconds (0.1), &CalcThroughput);
}

static void
StartMeasurement ()
{
  g_lastRx.resize (g_allSinks.size ());
  for (size_t i = 0; i < g_allSinks.size (); ++i)
    g_lastRx[i] = g_allSinks[i]->GetTotalRx ();
  for (uint32_t r = 0; r < NUM_REP; ++r)
    g_rep[r].lastTotalRx = g_allSinks[g_repSinkIdx[r]]->GetTotalRx ();
  Simulator::Schedule (Seconds (0.1), &CalcThroughput);
}

// ======================= connect socket traces =======================

static void
ConnectSocketTraces ()
{
  for (uint32_t r = 0; r < NUM_REP; ++r)
    {
      std::ostringstream cwndP, rttP, ifP;
      cwndP << "/NodeList/" << g_repNodeId[r]
            << "/$ns3::TcpL4Protocol/SocketList/0/CongestionWindow";
      rttP  << "/NodeList/" << g_repNodeId[r]
            << "/$ns3::TcpL4Protocol/SocketList/0/RTT";
      ifP   << "/NodeList/" << g_repNodeId[r]
            << "/$ns3::TcpL4Protocol/SocketList/0/BytesInFlight";

      Config::ConnectWithoutContext (cwndP.str (),
                                    MakeBoundCallback (&CwndTrace, r));
      Config::ConnectWithoutContext (rttP.str (),
                                    MakeBoundCallback (&RttTrace, r));
      Config::ConnectWithoutContext (ifP.str (),
                                    MakeBoundCallback (&InFlightTrace, r));
    }
  NS_LOG_UNCOND ("Socket traces connected for " << NUM_REP
                 << " representative flows");
}

// ===================================================================
//  main
// ===================================================================

int
main (int argc, char *argv[])
{
  uint32_t    nNodes    = 40;
  uint32_t    nFlows    = 20;
  uint32_t    pps       = 200;
  std::string transport = "cubic";

  CommandLine cmd (__FILE__);
  cmd.AddValue ("nNodes",    "Total nodes (incl. 2 routers)",  nNodes);
  cmd.AddValue ("nFlows",    "Number of TCP flows",            nFlows);
  cmd.AddValue ("pps",       "Packets per second per flow",    pps);
  cmd.AddValue ("transport", "TCP variant: cubic|reno|drnn",   transport);
  cmd.AddValue ("simTime",   "Simulation duration (s)",        g_simTime);
  cmd.AddValue ("port",      "OpenGym port (drnn only)",       g_gymPort);
  cmd.Parse (argc, argv);

  g_nFlows = nFlows;
  g_isDrnn = (transport == "drnn");

  uint32_t nSrc = (nNodes - 2) / 2;
  uint32_t nDst = nNodes - 2 - nSrc;
  NS_ABORT_MSG_IF (nSrc < 1 || nDst < 1,
                   "Need >= 4 nodes (2 routers + 1 src + 1 dst)");
  NS_ABORT_MSG_IF (nFlows < NUM_REP,
                   "Need >= 3 flows for representative tracing");

  // ---------- TCP variant ----------
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

  // ---------- TCP knobs ----------
  Config::SetDefault ("ns3::TcpSocket::SegmentSize",  UintegerValue (1024));
  Config::SetDefault ("ns3::TcpSocket::SndBufSize",   UintegerValue (1 << 20));
  Config::SetDefault ("ns3::TcpSocket::RcvBufSize",   UintegerValue (1 << 20));
  Config::SetDefault ("ns3::TcpSocket::InitialSlowStartThreshold",
                      UintegerValue (10 * 1024 * 1024));
  Config::SetDefault ("ns3::DropTailQueue<Packet>::MaxSize",
                      QueueSizeValue (QueueSize ("1p")));

  // ======================= nodes =======================
  NodeContainer srcNodes;  srcNodes.Create (nSrc);
  NodeContainer r1;        r1.Create (1);
  NodeContainer r2;        r2.Create (1);
  NodeContainer dstNodes;  dstNodes.Create (nDst);

  // Pick 3 representative flows (first, third, two-thirds)
  uint32_t repIdx[NUM_REP] = {0, nFlows / 3, 2 * nFlows / 3};
  for (uint32_t r = 0; r < NUM_REP; ++r)
    {
      g_repSinkIdx[r] = repIdx[r];
      g_repNodeId[r]  = srcNodes.Get (repIdx[r] % nSrc)->GetId ();
    }

  // ======================= links =======================
  PointToPointHelper p2pAccess;
  p2pAccess.SetDeviceAttribute  ("DataRate", StringValue ("100Mbps"));
  p2pAccess.SetChannelAttribute ("Delay",    StringValue ("2ms"));

  PointToPointHelper p2pBn;
  p2pBn.SetDeviceAttribute  ("DataRate", StringValue ("10Mbps"));
  p2pBn.SetChannelAttribute ("Delay",    StringValue ("20ms"));

  // Bottleneck R1 <-> R2
  NetDeviceContainer bnDev = p2pBn.Install (r1.Get (0), r2.Get (0));

  // Source access links (Src_i <-> R1)
  std::vector<NetDeviceContainer> srcDevs (nSrc);
  for (uint32_t i = 0; i < nSrc; ++i)
    srcDevs[i] = p2pAccess.Install (srcNodes.Get (i), r1.Get (0));

  // Destination access links (R2 <-> Dst_j)
  std::vector<NetDeviceContainer> dstDevs (nDst);
  for (uint32_t j = 0; j < nDst; ++j)
    dstDevs[j] = p2pAccess.Install (r2.Get (0), dstNodes.Get (j));

  // ======================= internet stack =======================
  InternetStackHelper internet;
  internet.Install (srcNodes);
  internet.Install (r1);
  internet.Install (r2);
  internet.Install (dstNodes);

  // ======================= traffic control =======================
  // Install BEFORE Ipv4AddressHelper::Assign()

  // Forward direction R1->R2 (data packets, 64-pkt queue)
  TrafficControlHelper tchFwd;
  tchFwd.SetRootQueueDisc ("ns3::FifoQueueDisc",
                            "MaxSize", StringValue ("64p"));
  QueueDiscContainer qdFwd = tchFwd.Install (bnDev.Get (0));
  qdFwd.Get (0)->TraceConnectWithoutContext ("Drop",
                                             MakeCallback (&DropTraceL1));

  // Reverse direction R2->R1 (ACK packets, 64-pkt queue)
  TrafficControlHelper tchRev;
  tchRev.SetRootQueueDisc ("ns3::FifoQueueDisc",
                            "MaxSize", StringValue ("64p"));
  QueueDiscContainer qdRev = tchRev.Install (bnDev.Get (1));
  qdRev.Get (0)->TraceConnectWithoutContext ("Drop",
                                             MakeCallback (&DropTraceL2));

  // ======================= IP addresses =======================
  Ipv4AddressHelper ipv4;

  // Bottleneck: 10.1.1.0/24
  ipv4.SetBase ("10.1.1.0", "255.255.255.0");
  ipv4.Assign (bnDev);

  // Source access: 10.1.(10+i).0/24
  for (uint32_t i = 0; i < nSrc; ++i)
    {
      std::ostringstream base;
      base << "10.1." << (i + 10) << ".0";
      ipv4.SetBase (base.str ().c_str (), "255.255.255.0");
      ipv4.Assign (srcDevs[i]);
    }

  // Destination access: 10.2.(10+j).0/24
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

  // ======================= TCP applications =======================
  uint16_t basePort = 9000;

  // Data rate per flow: pps * 1024-byte packets
  uint64_t dataRateBps = (uint64_t)pps * 1024 * 8;
  std::ostringstream drStr;
  drStr << dataRateBps << "bps";

  for (uint32_t i = 0; i < nFlows; ++i)
    {
      uint16_t port   = basePort + i;
      uint32_t srcIdx = i % nSrc;
      uint32_t dstIdx = i % nDst;

      // PacketSink on destination
      PacketSinkHelper sinkHelper ("ns3::TcpSocketFactory",
          InetSocketAddress (Ipv4Address::GetAny (), port));
      ApplicationContainer sinkApp = sinkHelper.Install (dstNodes.Get (dstIdx));
      sinkApp.Start (Seconds (0.0));
      sinkApp.Stop  (Seconds (g_simTime));
      g_allSinks.push_back (DynamicCast<PacketSink> (sinkApp.Get (0)));

      // OnOff TCP source (always-on, rate = pps * 1024 bytes/s)
      OnOffHelper onoff ("ns3::TcpSocketFactory",
          InetSocketAddress (dstAddr[dstIdx], port));
      onoff.SetAttribute ("DataRate",   StringValue (drStr.str ()));
      onoff.SetAttribute ("PacketSize", UintegerValue (1024));
      onoff.SetAttribute ("OnTime",  StringValue (
          "ns3::ConstantRandomVariable[Constant=1000000]"));
      onoff.SetAttribute ("OffTime", StringValue (
          "ns3::ConstantRandomVariable[Constant=0]"));

      ApplicationContainer srcApp = onoff.Install (srcNodes.Get (srcIdx));
      srcApp.Start (Seconds (0.5 + i * 0.01));
      srcApp.Stop  (Seconds (g_simTime));
    }

  // ======================= FlowMonitor =======================
  FlowMonitorHelper flowMonHelper;
  Ptr<FlowMonitor> flowMon = flowMonHelper.InstallAll ();

  // ======================= CSV files =======================
  std::string pfx = "wd_" + transport + "_";

  g_tputFile.open (pfx + "throughput.csv");
  g_tputFile << "Time,Total";
  for (uint32_t r = 0; r < NUM_REP; ++r) g_tputFile << ",Flow" << r;
  g_tputFile << ",Drops_L1,Drops_L2\n";

  g_cwndFile.open (pfx + "cwnd.csv");
  g_cwndFile << "Time";
  for (uint32_t r = 0; r < NUM_REP; ++r) g_cwndFile << ",Flow" << r << "_CWND";
  g_cwndFile << "\n";

  g_rttFile.open (pfx + "rtt.csv");
  g_rttFile << "Time";
  for (uint32_t r = 0; r < NUM_REP; ++r) g_rttFile << ",Flow" << r << "_RTT";
  g_rttFile << "\n";

  // ======================= schedule traces =======================
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
      // Only count forward (data) flows — destination port in our range
      if (t.destinationPort >= basePort
          && t.destinationPort < basePort + nFlows)
        {
          totalTxPkts   += it->second.txPackets;
          totalRxPkts   += it->second.rxPackets;
          totalLostPkts += it->second.lostPackets;
          totalRxBytes  += it->second.rxBytes;
          if (it->second.rxPackets > 0)
            {
              totalDelayMs  += it->second.delaySum.GetSeconds () * 1000.0;
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

  Simulator::Destroy ();

  // Human-readable
  std::cout << "\n=== Wired Dumbbell Simulation Complete ===\n"
    << "Transport:    " << transport << "\n"
    << "Nodes:        " << nNodes << " (" << nSrc << " src + 2 routers + "
                        << nDst << " dst)\n"
    << "Flows:        " << nFlows << "\n"
    << "PPS/flow:     " << pps << "\n"
    << "Duration:     " << g_simTime << " s\n"
    << "Throughput:   " << throughputMbps << " Mbps\n"
    << "Avg Delay:    " << avgDelayMs << " ms\n"
    << "PDR:          " << pdr << "\n"
    << "Drop Ratio:   " << dropRatio << "\n"
    << "Queue Drops:  " << g_dropsL1 << " (fwd) + " << g_dropsL2 << " (rev)\n"
    << "FlowMon Lost: " << totalLostPkts << "\n"
    << "CSV prefix:   " << pfx << "\n";

  // Machine-readable (shell script greps for this)
  std::cout << "SUMMARY,"
    << transport     << ","
    << nNodes        << ","
    << nFlows        << ","
    << pps           << ","
    << throughputMbps << ","
    << avgDelayMs    << ","
    << pdr           << ","
    << dropRatio     << ","
    << (g_dropsL1 + g_dropsL2) << ","
    << totalTxPkts   << ","
    << totalRxPkts   << "\n";

  return 0;
}

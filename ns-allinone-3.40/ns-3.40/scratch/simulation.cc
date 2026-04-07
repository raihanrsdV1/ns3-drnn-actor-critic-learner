/*
 * simulation.cc — Simple wired dumbbell TCP simulation (Cubic/Reno/DRNN)
 *
 * Topology:
 *
 *   Src0 ----\
 *             \ 100Mbps/2ms
 *   Src1 ----- R1 ====== R2 ----- Dst0
 *                 5Mbps/20ms        100Mbps/2ms
 *                     (bottleneck)
 *                               \-- Dst1
 *
 * DRNN mode integrates with scratch/drnn_agent_cont.py through ns3-gym.
 * Observation: [cwnd,rtt,bif] per flow + [drops_fwd, drops_rev]
 * Action: one float (target cwnd bytes) applied to both flows.
 */

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/applications-module.h"
#include "ns3/traffic-control-module.h"
#include "ns3/opengym-module.h"
#include <fstream>
#include <sstream>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("SimpleDumbbellSim");

class TcpDumbbellDrnn : public TcpCongestionOps
{
public:
  static TypeId GetTypeId (void);
  TcpDumbbellDrnn ();
  TcpDumbbellDrnn (const TcpDumbbellDrnn &sock);
  ~TcpDumbbellDrnn () override;

  std::string GetName () const override { return "TcpDumbbellDrnn"; }

  uint32_t GetSsThresh (Ptr<const TcpSocketState> tcb,
                        uint32_t bytesInFlight) override;
  void IncreaseWindow (Ptr<TcpSocketState> tcb,
                       uint32_t segmentsAcked) override;
  Ptr<TcpCongestionOps> Fork () override;

  static float g_target_cwnd;
};

static const uint32_t DRNN_MIN_CWND = 2 * 1448;
static const uint32_t DRNN_MAX_CWND = 144 * 1024;
float TcpDumbbellDrnn::g_target_cwnd = static_cast<float> (12 * 1024);

NS_OBJECT_ENSURE_REGISTERED (TcpDumbbellDrnn);

TypeId
TcpDumbbellDrnn::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::TcpDumbbellDrnn")
    .SetParent<TcpCongestionOps> ()
    .SetGroupName ("Internet")
    .AddConstructor<TcpDumbbellDrnn> ();
  return tid;
}

TcpDumbbellDrnn::TcpDumbbellDrnn () : TcpCongestionOps () {}
TcpDumbbellDrnn::TcpDumbbellDrnn (const TcpDumbbellDrnn &s) : TcpCongestionOps (s) {}
TcpDumbbellDrnn::~TcpDumbbellDrnn () {}

uint32_t
TcpDumbbellDrnn::GetSsThresh (Ptr<const TcpSocketState> tcb,
                              uint32_t /*bytesInFlight*/)
{
  return std::max<uint32_t> (DRNN_MIN_CWND, tcb->m_cWnd.Get () / 2);
}

void
TcpDumbbellDrnn::IncreaseWindow (Ptr<TcpSocketState> tcb, uint32_t segmentsAcked)
{
  if (segmentsAcked == 0)
    {
      return;
    }

  uint32_t seg = tcb->m_segmentSize;
  uint32_t target = static_cast<uint32_t> (
      std::max ((float) DRNN_MIN_CWND,
                std::min ((float) DRNN_MAX_CWND, g_target_cwnd)));
  target = std::max (2 * seg, (target / seg) * seg);
  tcb->m_cWnd = target;
}

Ptr<TcpCongestionOps>
TcpDumbbellDrnn::Fork ()
{
  return CopyObject<TcpDumbbellDrnn> (this);
}

static const uint32_t NUM_FLOWS = 2;
static const double   DEFAULT_SIM_TIME = 60.0;

struct FlowState
{
  uint64_t lastTotalRx = 0;
  uint32_t cwnd = 0;
  double rttMs = 0.0;
  uint32_t bytesInFlight = 0;
};

static FlowState g_flow[NUM_FLOWS];
static Ptr<PacketSink> g_sink[NUM_FLOWS];
static uint32_t g_srcNodeId[NUM_FLOWS];

static uint32_t g_dropsFwd = 0;
static uint32_t g_dropsRev = 0;
static uint32_t g_lastDropsFwd = 0;
static uint32_t g_lastDropsRev = 0;
static uint32_t g_gymLastDropsFwd = 0;
static uint32_t g_gymLastDropsRev = 0;

static std::ofstream g_tputFile;
static std::ofstream g_cwndFile;
static std::ofstream g_rttFile;

static bool g_isDrnn = false;
static double g_simTime = DEFAULT_SIM_TIME;

static Ptr<OpenGymInterface> g_openGym;
static uint32_t g_gymPort = 5557;
static const uint32_t OBS_DIM = NUM_FLOWS * 3 + 2;

static void
CwndTrace (uint32_t flowId, uint32_t /*oldVal*/, uint32_t newVal)
{
  g_flow[flowId].cwnd = newVal;
}

static void
RttTrace (uint32_t flowId, Time /*oldVal*/, Time newVal)
{
  g_flow[flowId].rttMs = newVal.GetMilliSeconds ();
}

static void
InFlightTrace (uint32_t flowId, uint32_t /*oldVal*/, uint32_t newVal)
{
  g_flow[flowId].bytesInFlight = newVal;
}

static void
DropTraceFwd (Ptr<const QueueDiscItem> /*item*/)
{
  ++g_dropsFwd;
}

static void
DropTraceRev (Ptr<const QueueDiscItem> /*item*/)
{
  ++g_dropsRev;
}

Ptr<OpenGymSpace>
GetObsSpace ()
{
  std::vector<uint32_t> shape = {OBS_DIM};
  return CreateObject<OpenGymBoxSpace> (0.0f, 1e9f, shape, TypeNameGet<float> ());
}

Ptr<OpenGymSpace>
GetActSpace ()
{
  std::vector<uint32_t> shape = {1};
  return CreateObject<OpenGymBoxSpace> ((float) DRNN_MIN_CWND,
                                        (float) DRNN_MAX_CWND,
                                        shape,
                                        TypeNameGet<float> ());
}

Ptr<OpenGymDataContainer>
GetObs ()
{
  Ptr<OpenGymBoxContainer<float>> box = CreateObject<OpenGymBoxContainer<float>> ();
  for (uint32_t i = 0; i < NUM_FLOWS; ++i)
    {
      box->AddValue ((float) g_flow[i].cwnd);
      box->AddValue ((float) g_flow[i].rttMs);
      box->AddValue ((float) g_flow[i].bytesInFlight);
    }

  uint32_t dFwd = g_dropsFwd - g_gymLastDropsFwd;
  uint32_t dRev = g_dropsRev - g_gymLastDropsRev;
  g_gymLastDropsFwd = g_dropsFwd;
  g_gymLastDropsRev = g_dropsRev;

  box->AddValue ((float) dFwd);
  box->AddValue ((float) dRev);
  return box;
}

float
GetReward ()
{
  return 0.0f; // Python agent computes reward.
}

bool
GetDone ()
{
  return Simulator::Now ().GetSeconds () >= g_simTime;
}

std::string
GetInfo ()
{
  return "";
}

bool
ExecuteAction (Ptr<OpenGymDataContainer> action)
{
  Ptr<OpenGymBoxContainer<float>> box = DynamicCast<OpenGymBoxContainer<float>> (action);
  if (box && !box->GetData ().empty ())
    {
      float val = box->GetData ()[0];
      TcpDumbbellDrnn::g_target_cwnd = std::max ((float) DRNN_MIN_CWND,
                                                 std::min ((float) DRNN_MAX_CWND, val));
    }
  return true;
}

static void
DrnnStep (double dt)
{
  g_openGym->NotifyCurrentState ();
  if (!GetDone ())
    {
      Simulator::Schedule (Seconds (dt), &DrnnStep, dt);
    }
}

static void
CalcThroughput ()
{
  double now = Simulator::Now ().GetSeconds ();
  if (now >= g_simTime)
    {
      return;
    }

  double flowTput[NUM_FLOWS];
  double total = 0.0;

  for (uint32_t i = 0; i < NUM_FLOWS; ++i)
    {
      uint64_t rx = g_sink[i]->GetTotalRx ();
      double mbps = (rx - g_flow[i].lastTotalRx) * 8.0 / 0.1 / 1e6;
      g_flow[i].lastTotalRx = rx;
      flowTput[i] = mbps;
      total += mbps;
    }

  uint32_t dFwd = g_dropsFwd - g_lastDropsFwd;
  uint32_t dRev = g_dropsRev - g_lastDropsRev;
  g_lastDropsFwd = g_dropsFwd;
  g_lastDropsRev = g_dropsRev;

  g_tputFile << now << "," << total;
  for (uint32_t i = 0; i < NUM_FLOWS; ++i)
    {
      g_tputFile << "," << flowTput[i];
    }
  g_tputFile << "," << dFwd << "," << dRev << "\n";

  g_cwndFile << now;
  for (uint32_t i = 0; i < NUM_FLOWS; ++i)
    {
      g_cwndFile << "," << g_flow[i].cwnd;
    }
  g_cwndFile << "\n";

  g_rttFile << now;
  for (uint32_t i = 0; i < NUM_FLOWS; ++i)
    {
      g_rttFile << "," << g_flow[i].rttMs;
    }
  g_rttFile << "\n";

  Simulator::Schedule (Seconds (0.1), &CalcThroughput);
}

static void
StartMeasurement ()
{
  for (uint32_t i = 0; i < NUM_FLOWS; ++i)
    {
      g_flow[i].lastTotalRx = g_sink[i]->GetTotalRx ();
    }
  Simulator::Schedule (Seconds (0.1), &CalcThroughput);
}

static void
ConnectSocketTraces ()
{
  for (uint32_t i = 0; i < NUM_FLOWS; ++i)
    {
      std::ostringstream cwndPath, rttPath, ifPath;
      cwndPath << "/NodeList/" << g_srcNodeId[i]
               << "/$ns3::TcpL4Protocol/SocketList/0/CongestionWindow";
      rttPath  << "/NodeList/" << g_srcNodeId[i]
               << "/$ns3::TcpL4Protocol/SocketList/0/RTT";
      ifPath   << "/NodeList/" << g_srcNodeId[i]
               << "/$ns3::TcpL4Protocol/SocketList/0/BytesInFlight";

      Config::ConnectWithoutContext (cwndPath.str (), MakeBoundCallback (&CwndTrace, i));
      Config::ConnectWithoutContext (rttPath.str (),  MakeBoundCallback (&RttTrace, i));
      Config::ConnectWithoutContext (ifPath.str (),   MakeBoundCallback (&InFlightTrace, i));
    }
}

int
main (int argc, char *argv[])
{
  std::string transport = "cubic";

  CommandLine cmd (__FILE__);
  cmd.AddValue ("transport", "TCP variant: cubic | reno | drnn", transport);
  cmd.AddValue ("simTime", "Simulation duration in seconds", g_simTime);
  cmd.AddValue ("port", "OpenGym port (drnn mode)", g_gymPort);
  cmd.Parse (argc, argv);

  g_isDrnn = (transport == "drnn");

  if (transport == "cubic")
    {
      Config::SetDefault ("ns3::TcpL4Protocol::SocketType",
                          TypeIdValue (TcpCubic::GetTypeId ()));
    }
  else if (transport == "reno")
    {
      Config::SetDefault ("ns3::TcpL4Protocol::SocketType",
                          TypeIdValue (TcpNewReno::GetTypeId ()));
    }
  else if (transport == "drnn")
    {
      Config::SetDefault ("ns3::TcpL4Protocol::SocketType",
                          TypeIdValue (TcpDumbbellDrnn::GetTypeId ()));
    }
  else
    {
      NS_FATAL_ERROR ("Unknown transport: " << transport);
    }

  Config::SetDefault ("ns3::TcpSocket::SegmentSize", UintegerValue (1448));
  Config::SetDefault ("ns3::DropTailQueue<Packet>::MaxSize",
                      QueueSizeValue (QueueSize ("1p")));

  NodeContainer srcNodes;
  srcNodes.Create (NUM_FLOWS);
  NodeContainer r1Node;
  r1Node.Create (1);
  NodeContainer r2Node;
  r2Node.Create (1);
  NodeContainer dstNodes;
  dstNodes.Create (NUM_FLOWS);

  for (uint32_t i = 0; i < NUM_FLOWS; ++i)
    {
      g_srcNodeId[i] = srcNodes.Get (i)->GetId ();
    }

  PointToPointHelper p2pEdge;
  p2pEdge.SetDeviceAttribute ("DataRate", StringValue ("100Mbps"));
  p2pEdge.SetChannelAttribute ("Delay", StringValue ("2ms"));

  PointToPointHelper p2pBottle;
  p2pBottle.SetDeviceAttribute ("DataRate", StringValue ("5Mbps"));
  p2pBottle.SetChannelAttribute ("Delay", StringValue ("20ms"));

  NetDeviceContainer srcR1Dev[NUM_FLOWS];
  NetDeviceContainer r2DstDev[NUM_FLOWS];

  for (uint32_t i = 0; i < NUM_FLOWS; ++i)
    {
      srcR1Dev[i] = p2pEdge.Install (srcNodes.Get (i), r1Node.Get (0));
      r2DstDev[i] = p2pEdge.Install (r2Node.Get (0), dstNodes.Get (i));
    }

  NetDeviceContainer bnDev = p2pBottle.Install (r1Node.Get (0), r2Node.Get (0));

  InternetStackHelper internet;
  internet.Install (srcNodes);
  internet.Install (r1Node);
  internet.Install (r2Node);
  internet.Install (dstNodes);

  TrafficControlHelper tch;
  tch.SetRootQueueDisc ("ns3::FifoQueueDisc", "MaxSize", StringValue ("40p"));
  QueueDiscContainer qdFwd = tch.Install (bnDev.Get (0));
  QueueDiscContainer qdRev = tch.Install (bnDev.Get (1));
  qdFwd.Get (0)->TraceConnectWithoutContext ("Drop", MakeCallback (&DropTraceFwd));
  qdRev.Get (0)->TraceConnectWithoutContext ("Drop", MakeCallback (&DropTraceRev));

  Ipv4AddressHelper ipv4;

  Ipv4InterfaceContainer srcIf[NUM_FLOWS];
  Ipv4InterfaceContainer dstIf[NUM_FLOWS];

  for (uint32_t i = 0; i < NUM_FLOWS; ++i)
    {
      std::ostringstream net;
      net << "10.1." << (i + 1) << ".0";
      ipv4.SetBase (net.str ().c_str (), "255.255.255.0");
      srcIf[i] = ipv4.Assign (srcR1Dev[i]);
    }

  ipv4.SetBase ("10.1.10.0", "255.255.255.0");
  ipv4.Assign (bnDev);

  for (uint32_t i = 0; i < NUM_FLOWS; ++i)
    {
      std::ostringstream net;
      net << "10.1." << (20 + i) << ".0";
      ipv4.SetBase (net.str ().c_str (), "255.255.255.0");
      dstIf[i] = ipv4.Assign (r2DstDev[i]);
    }

  Ipv4GlobalRoutingHelper::PopulateRoutingTables ();

  uint16_t basePort = 9000;
  for (uint32_t i = 0; i < NUM_FLOWS; ++i)
    {
      uint16_t port = basePort + i;
      PacketSinkHelper sinkHelper ("ns3::TcpSocketFactory",
                                   InetSocketAddress (Ipv4Address::GetAny (), port));
      ApplicationContainer sinkApp = sinkHelper.Install (dstNodes.Get (i));
      sinkApp.Start (Seconds (0.0));
      sinkApp.Stop (Seconds (g_simTime));
      g_sink[i] = DynamicCast<PacketSink> (sinkApp.Get (0));

      BulkSendHelper srcHelper ("ns3::TcpSocketFactory",
                                InetSocketAddress (dstIf[i].GetAddress (1), port));
      srcHelper.SetAttribute ("MaxBytes", UintegerValue (0));
      srcHelper.SetAttribute ("SendSize", UintegerValue (1448));
      ApplicationContainer srcApp = srcHelper.Install (srcNodes.Get (i));
      srcApp.Start (Seconds (0.5 + 0.2 * i));
      srcApp.Stop (Seconds (g_simTime));
    }

  std::string prefix = "db_" + transport + "_";

  g_tputFile.open (prefix + "throughput.csv");
  g_tputFile << "Time,Total";
  for (uint32_t i = 0; i < NUM_FLOWS; ++i)
    {
      g_tputFile << ",Flow" << i;
    }
  g_tputFile << ",Drops_L1,Drops_L2\n";

  g_cwndFile.open (prefix + "cwnd.csv");
  g_cwndFile << "Time";
  for (uint32_t i = 0; i < NUM_FLOWS; ++i)
    {
      g_cwndFile << ",Flow" << i << "_CWND";
    }
  g_cwndFile << "\n";

  g_rttFile.open (prefix + "rtt.csv");
  g_rttFile << "Time";
  for (uint32_t i = 0; i < NUM_FLOWS; ++i)
    {
      g_rttFile << ",Flow" << i << "_RTT";
    }
  g_rttFile << "\n";

  Simulator::Schedule (Seconds (1.3), &ConnectSocketTraces);
  Simulator::Schedule (Seconds (1.4), &StartMeasurement);

  if (g_isDrnn)
    {
      g_openGym = CreateObject<OpenGymInterface> (g_gymPort);
      g_openGym->SetGetObservationSpaceCb (MakeCallback (&GetObsSpace));
      g_openGym->SetGetActionSpaceCb (MakeCallback (&GetActSpace));
      g_openGym->SetGetObservationCb (MakeCallback (&GetObs));
      g_openGym->SetGetRewardCb (MakeCallback (&GetReward));
      g_openGym->SetGetGameOverCb (MakeCallback (&GetDone));
      g_openGym->SetGetExtraInfoCb (MakeCallback (&GetInfo));
      g_openGym->SetExecuteActionsCb (MakeCallback (&ExecuteAction));
      Simulator::Schedule (Seconds (1.8), &DrnnStep, 0.1);
    }

  Simulator::Stop (Seconds (g_simTime));
  Simulator::Run ();

  g_tputFile.close ();
  g_cwndFile.close ();
  g_rttFile.close ();

  if (g_isDrnn && g_openGym)
    {
      g_openGym->NotifySimulationEnd ();
    }

  Simulator::Destroy ();

  std::cout << "\n=== Simple Dumbbell Simulation Complete ===\n"
            << "Transport:   " << transport << "\n"
            << "Duration:    " << g_simTime << " s\n"
            << "Flows:       " << NUM_FLOWS << "\n"
            << "Drops fwd:   " << g_dropsFwd << "\n"
            << "Drops rev:   " << g_dropsRev << "\n"
            << "CSV prefix:  " << prefix << "\n";

  return 0;
}

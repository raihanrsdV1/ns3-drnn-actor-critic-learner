/*
 * multi-bottleneck-sim.cc — Challenging multi-bottleneck wired topology
 *
 * Topology:
 *
 *   Src0 --\
 *   Src1 ---\
 *   Src2 ---- R1 ==(10Mbps/15ms,50p)== R2 ==(6Mbps/25ms,30p)== R3 ---- Dst0
 *   Src3 ---/                                              |               Dst1
 *                                                          +---- Dst2
 *                                                          +---- Dst3
 *
 * Extra cross-traffic (UDP bursts):
 *   CrossA -> R1 -> ... -> Dst0   (hits both bottlenecks)
 *   CrossB -> R2 -> ... -> Dst1   (hits second bottleneck)
 *
 * DRNN mode uses ns3-gym and integrates with scratch/drnn_agent_cont.py.
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

NS_LOG_COMPONENT_DEFINE ("MultiBottleneckSim");

class TcpMultiBottleneckDrnn : public TcpCongestionOps
{
public:
  static TypeId GetTypeId (void);
  TcpMultiBottleneckDrnn ();
  TcpMultiBottleneckDrnn (const TcpMultiBottleneckDrnn &sock);
  ~TcpMultiBottleneckDrnn () override;

  std::string GetName () const override { return "TcpMultiBottleneckDrnn"; }

  uint32_t GetSsThresh (Ptr<const TcpSocketState> tcb,
                        uint32_t bytesInFlight) override;
  void IncreaseWindow (Ptr<TcpSocketState> tcb,
                       uint32_t segmentsAcked) override;
  Ptr<TcpCongestionOps> Fork () override;

  static float g_target_cwnd;
};

static const uint32_t DRNN_MIN_CWND = 2 * 1448;
static const uint32_t DRNN_MAX_CWND = 192 * 1024;
float TcpMultiBottleneckDrnn::g_target_cwnd = static_cast<float> (16 * 1024);

NS_OBJECT_ENSURE_REGISTERED (TcpMultiBottleneckDrnn);

TypeId
TcpMultiBottleneckDrnn::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::TcpMultiBottleneckDrnn")
    .SetParent<TcpCongestionOps> ()
    .SetGroupName ("Internet")
    .AddConstructor<TcpMultiBottleneckDrnn> ();
  return tid;
}

TcpMultiBottleneckDrnn::TcpMultiBottleneckDrnn () : TcpCongestionOps () {}
TcpMultiBottleneckDrnn::TcpMultiBottleneckDrnn (const TcpMultiBottleneckDrnn &s) : TcpCongestionOps (s) {}
TcpMultiBottleneckDrnn::~TcpMultiBottleneckDrnn () {}

uint32_t
TcpMultiBottleneckDrnn::GetSsThresh (Ptr<const TcpSocketState> tcb,
                                     uint32_t /*bytesInFlight*/)
{
  return std::max<uint32_t> (DRNN_MIN_CWND, tcb->m_cWnd.Get () / 2);
}

void
TcpMultiBottleneckDrnn::IncreaseWindow (Ptr<TcpSocketState> tcb, uint32_t segmentsAcked)
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
TcpMultiBottleneckDrnn::Fork ()
{
  return CopyObject<TcpMultiBottleneckDrnn> (this);
}

static const uint32_t NUM_FLOWS = 4;

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

static uint32_t g_dropsL1 = 0;
static uint32_t g_dropsL2 = 0;
static uint32_t g_lastDropsL1 = 0;
static uint32_t g_lastDropsL2 = 0;
static uint32_t g_gymLastDropsL1 = 0;
static uint32_t g_gymLastDropsL2 = 0;

static std::ofstream g_tputFile;
static std::ofstream g_cwndFile;
static std::ofstream g_rttFile;

static bool g_isDrnn = false;
static double g_simTime = 70.0;

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
DropTraceL1 (Ptr<const QueueDiscItem> /*item*/)
{
  ++g_dropsL1;
}

static void
DropTraceL2 (Ptr<const QueueDiscItem> /*item*/)
{
  ++g_dropsL2;
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

  uint32_t d1 = g_dropsL1 - g_gymLastDropsL1;
  uint32_t d2 = g_dropsL2 - g_gymLastDropsL2;
  g_gymLastDropsL1 = g_dropsL1;
  g_gymLastDropsL2 = g_dropsL2;

  box->AddValue ((float) d1);
  box->AddValue ((float) d2);
  return box;
}

float
GetReward ()
{
  return 0.0f;
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
      TcpMultiBottleneckDrnn::g_target_cwnd = std::max ((float) DRNN_MIN_CWND,
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

  uint32_t d1 = g_dropsL1 - g_lastDropsL1;
  uint32_t d2 = g_dropsL2 - g_lastDropsL2;
  g_lastDropsL1 = g_dropsL1;
  g_lastDropsL2 = g_dropsL2;

  g_tputFile << now << "," << total;
  for (uint32_t i = 0; i < NUM_FLOWS; ++i)
    {
      g_tputFile << "," << flowTput[i];
    }
  g_tputFile << "," << d1 << "," << d2 << "\n";

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
      rttPath << "/NodeList/" << g_srcNodeId[i]
              << "/$ns3::TcpL4Protocol/SocketList/0/RTT";
      ifPath << "/NodeList/" << g_srcNodeId[i]
             << "/$ns3::TcpL4Protocol/SocketList/0/BytesInFlight";

      Config::ConnectWithoutContext (cwndPath.str (), MakeBoundCallback (&CwndTrace, i));
      Config::ConnectWithoutContext (rttPath.str (), MakeBoundCallback (&RttTrace, i));
      Config::ConnectWithoutContext (ifPath.str (), MakeBoundCallback (&InFlightTrace, i));
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
                          TypeIdValue (TcpMultiBottleneckDrnn::GetTypeId ()));
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
  NodeContainer r3Node;
  r3Node.Create (1);
  NodeContainer dstNodes;
  dstNodes.Create (NUM_FLOWS);
  NodeContainer crossA;
  crossA.Create (1);
  NodeContainer crossB;
  crossB.Create (1);

  for (uint32_t i = 0; i < NUM_FLOWS; ++i)
    {
      g_srcNodeId[i] = srcNodes.Get (i)->GetId ();
    }

  PointToPointHelper p2pEdge;
  p2pEdge.SetDeviceAttribute ("DataRate", StringValue ("100Mbps"));
  p2pEdge.SetChannelAttribute ("Delay", StringValue ("2ms"));

  PointToPointHelper p2pBn1;
  p2pBn1.SetDeviceAttribute ("DataRate", StringValue ("10Mbps"));
  p2pBn1.SetChannelAttribute ("Delay", StringValue ("15ms"));

  PointToPointHelper p2pBn2;
  p2pBn2.SetDeviceAttribute ("DataRate", StringValue ("6Mbps"));
  p2pBn2.SetChannelAttribute ("Delay", StringValue ("25ms"));

  PointToPointHelper p2pCross;
  p2pCross.SetDeviceAttribute ("DataRate", StringValue ("100Mbps"));
  p2pCross.SetChannelAttribute ("Delay", StringValue ("1ms"));

  NetDeviceContainer srcR1Dev[NUM_FLOWS];
  NetDeviceContainer r3DstDev[NUM_FLOWS];
  for (uint32_t i = 0; i < NUM_FLOWS; ++i)
    {
      srcR1Dev[i] = p2pEdge.Install (srcNodes.Get (i), r1Node.Get (0));
      r3DstDev[i] = p2pEdge.Install (r3Node.Get (0), dstNodes.Get (i));
    }

  NetDeviceContainer bn1Dev = p2pBn1.Install (r1Node.Get (0), r2Node.Get (0));
  NetDeviceContainer bn2Dev = p2pBn2.Install (r2Node.Get (0), r3Node.Get (0));
  NetDeviceContainer crossADev = p2pCross.Install (crossA.Get (0), r1Node.Get (0));
  NetDeviceContainer crossBDev = p2pCross.Install (crossB.Get (0), r2Node.Get (0));

  InternetStackHelper internet;
  internet.Install (srcNodes);
  internet.Install (r1Node);
  internet.Install (r2Node);
  internet.Install (r3Node);
  internet.Install (dstNodes);
  internet.Install (crossA);
  internet.Install (crossB);

  TrafficControlHelper tch1;
  tch1.SetRootQueueDisc ("ns3::FifoQueueDisc", "MaxSize", StringValue ("50p"));
  QueueDiscContainer qd1 = tch1.Install (bn1Dev.Get (0));
  qd1.Get (0)->TraceConnectWithoutContext ("Drop", MakeCallback (&DropTraceL1));

  TrafficControlHelper tch2;
  tch2.SetRootQueueDisc ("ns3::FifoQueueDisc", "MaxSize", StringValue ("30p"));
  QueueDiscContainer qd2 = tch2.Install (bn2Dev.Get (0));
  qd2.Get (0)->TraceConnectWithoutContext ("Drop", MakeCallback (&DropTraceL2));

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

  ipv4.SetBase ("10.1.20.0", "255.255.255.0");
  ipv4.Assign (bn1Dev);

  ipv4.SetBase ("10.1.21.0", "255.255.255.0");
  ipv4.Assign (bn2Dev);

  for (uint32_t i = 0; i < NUM_FLOWS; ++i)
    {
      std::ostringstream net;
      net << "10.1." << (30 + i) << ".0";
      ipv4.SetBase (net.str ().c_str (), "255.255.255.0");
      dstIf[i] = ipv4.Assign (r3DstDev[i]);
    }

  ipv4.SetBase ("10.1.40.0", "255.255.255.0");
  ipv4.Assign (crossADev);

  ipv4.SetBase ("10.1.41.0", "255.255.255.0");
  ipv4.Assign (crossBDev);

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

  uint16_t udpPortA = 11000;
  uint16_t udpPortB = 11001;

  PacketSinkHelper udpSinkA ("ns3::UdpSocketFactory",
                             InetSocketAddress (Ipv4Address::GetAny (), udpPortA));
  ApplicationContainer udpSinkAppA = udpSinkA.Install (dstNodes.Get (0));
  udpSinkAppA.Start (Seconds (0.0));
  udpSinkAppA.Stop (Seconds (g_simTime));

  PacketSinkHelper udpSinkB ("ns3::UdpSocketFactory",
                             InetSocketAddress (Ipv4Address::GetAny (), udpPortB));
  ApplicationContainer udpSinkAppB = udpSinkB.Install (dstNodes.Get (1));
  udpSinkAppB.Start (Seconds (0.0));
  udpSinkAppB.Stop (Seconds (g_simTime));

  OnOffHelper onoffA ("ns3::UdpSocketFactory",
                      InetSocketAddress (dstIf[0].GetAddress (1), udpPortA));
  onoffA.SetAttribute ("DataRate", StringValue ("4Mbps"));
  onoffA.SetAttribute ("PacketSize", UintegerValue (512));
  onoffA.SetAttribute ("OnTime", StringValue ("ns3::ConstantRandomVariable[Constant=0.4]"));
  onoffA.SetAttribute ("OffTime", StringValue ("ns3::ConstantRandomVariable[Constant=0.4]"));
  ApplicationContainer appA = onoffA.Install (crossA.Get (0));
  appA.Start (Seconds (2.0));
  appA.Stop (Seconds (g_simTime));

  OnOffHelper onoffB ("ns3::UdpSocketFactory",
                      InetSocketAddress (dstIf[1].GetAddress (1), udpPortB));
  onoffB.SetAttribute ("DataRate", StringValue ("3Mbps"));
  onoffB.SetAttribute ("PacketSize", UintegerValue (512));
  onoffB.SetAttribute ("OnTime", StringValue ("ns3::ConstantRandomVariable[Constant=0.6]"));
  onoffB.SetAttribute ("OffTime", StringValue ("ns3::ConstantRandomVariable[Constant=0.6]"));
  ApplicationContainer appB = onoffB.Install (crossB.Get (0));
  appB.Start (Seconds (3.0));
  appB.Stop (Seconds (g_simTime));

  std::string prefix = "mb_" + transport + "_";

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

  Simulator::Schedule (Seconds (1.8), &ConnectSocketTraces);
  Simulator::Schedule (Seconds (2.0), &StartMeasurement);

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
      Simulator::Schedule (Seconds (2.2), &DrnnStep, 0.1);
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

  std::cout << "\n=== Multi-Bottleneck Simulation Complete ===\n"
            << "Transport:   " << transport << "\n"
            << "Duration:    " << g_simTime << " s\n"
            << "Flows:       " << NUM_FLOWS << "\n"
            << "Drops L1:    " << g_dropsL1 << "\n"
            << "Drops L2:    " << g_dropsL2 << "\n"
            << "CSV prefix:  " << prefix << "\n";

  return 0;
}

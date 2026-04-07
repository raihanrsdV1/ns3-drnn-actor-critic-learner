/*
 * fat-tree-incast-sim.cc — Data-center style multi-tier topology (incast stress)
 *
 * Incast scenario:
 *   12 senders (servers) simultaneously send TCP data to 1 receiver.
 *   Traffic converges through a multi-tier path and a tiny final queue,
 *   creating severe micro-burst pressure and drop spikes.
 *
 * Topology (wired, simplified fat-tree-like):
 *
 *  sender(12) -> edge(3) -> agg(2) -> core(1) -> recvEdge(1) -> receiver(1)
 *
 * The final two links are constrained and queue-limited:
 *   core->recvEdge: 300 Mbps, 40p queue
 *   recvEdge->receiver: 100 Mbps, 20p queue   (primary incast drop point)
 *
 * DRNN mode integrates with scratch/drnn_agent_cont.py via ns3-gym.
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
#include <vector>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("FatTreeIncastSim");

class TcpFatTreeDrnn : public TcpCongestionOps
{
public:
  static TypeId GetTypeId (void);
  TcpFatTreeDrnn ();
  TcpFatTreeDrnn (const TcpFatTreeDrnn &sock);
  ~TcpFatTreeDrnn () override;

  std::string GetName () const override { return "TcpFatTreeDrnn"; }

  uint32_t GetSsThresh (Ptr<const TcpSocketState> tcb,
                        uint32_t bytesInFlight) override;
  void IncreaseWindow (Ptr<TcpSocketState> tcb,
                       uint32_t segmentsAcked) override;
  Ptr<TcpCongestionOps> Fork () override;

  static float g_target_cwnd;
};

static const uint32_t DRNN_MIN_CWND = 2 * 1448;
static const uint32_t DRNN_MAX_CWND = 512 * 1024;
float TcpFatTreeDrnn::g_target_cwnd = static_cast<float> (32 * 1024);

NS_OBJECT_ENSURE_REGISTERED (TcpFatTreeDrnn);

TypeId
TcpFatTreeDrnn::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::TcpFatTreeDrnn")
    .SetParent<TcpCongestionOps> ()
    .SetGroupName ("Internet")
    .AddConstructor<TcpFatTreeDrnn> ();
  return tid;
}

TcpFatTreeDrnn::TcpFatTreeDrnn () : TcpCongestionOps () {}
TcpFatTreeDrnn::TcpFatTreeDrnn (const TcpFatTreeDrnn &s) : TcpCongestionOps (s) {}
TcpFatTreeDrnn::~TcpFatTreeDrnn () {}

uint32_t
TcpFatTreeDrnn::GetSsThresh (Ptr<const TcpSocketState> tcb,
                             uint32_t /*bytesInFlight*/)
{
  return std::max<uint32_t> (DRNN_MIN_CWND, tcb->m_cWnd.Get () / 2);
}

void
TcpFatTreeDrnn::IncreaseWindow (Ptr<TcpSocketState> tcb, uint32_t segmentsAcked)
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
TcpFatTreeDrnn::Fork ()
{
  return CopyObject<TcpFatTreeDrnn> (this);
}

static const uint32_t NUM_SENDERS = 12;
static const uint32_t NUM_EDGES = 3;
static const uint32_t NUM_AGGS = 2;
static const uint32_t SENDERS_PER_EDGE = 4;

struct FlowState
{
  uint64_t lastTotalRx = 0;
  uint32_t cwnd = 0;
  double rttMs = 0.0;
  uint32_t bytesInFlight = 0;
};

static FlowState g_flow[NUM_SENDERS];
static Ptr<PacketSink> g_sink[NUM_SENDERS];
static uint32_t g_senderNodeId[NUM_SENDERS];

static uint32_t g_dropsCoreToRecvEdge = 0; // L1
static uint32_t g_dropsRecvEdgeToHost = 0; // L2
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
static const uint32_t OBS_DIM = NUM_SENDERS * 3 + 2;

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
  ++g_dropsCoreToRecvEdge;
}

static void
DropTraceL2 (Ptr<const QueueDiscItem> /*item*/)
{
  ++g_dropsRecvEdgeToHost;
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
  for (uint32_t i = 0; i < NUM_SENDERS; ++i)
    {
      box->AddValue ((float) g_flow[i].cwnd);
      box->AddValue ((float) g_flow[i].rttMs);
      box->AddValue ((float) g_flow[i].bytesInFlight);
    }

  uint32_t d1 = g_dropsCoreToRecvEdge - g_gymLastDropsL1;
  uint32_t d2 = g_dropsRecvEdgeToHost - g_gymLastDropsL2;
  g_gymLastDropsL1 = g_dropsCoreToRecvEdge;
  g_gymLastDropsL2 = g_dropsRecvEdgeToHost;

  box->AddValue ((float) d1);
  box->AddValue ((float) d2);
  return box;
}

float
GetReward ()
{
  return 0.0f; // reward is computed in Python agent
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
      TcpFatTreeDrnn::g_target_cwnd = std::max ((float) DRNN_MIN_CWND,
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

  double flowTput[NUM_SENDERS];
  double total = 0.0;

  for (uint32_t i = 0; i < NUM_SENDERS; ++i)
    {
      uint64_t rx = g_sink[i]->GetTotalRx ();
      double mbps = (rx - g_flow[i].lastTotalRx) * 8.0 / 0.1 / 1e6;
      g_flow[i].lastTotalRx = rx;
      flowTput[i] = mbps;
      total += mbps;
    }

  uint32_t d1 = g_dropsCoreToRecvEdge - g_lastDropsL1;
  uint32_t d2 = g_dropsRecvEdgeToHost - g_lastDropsL2;
  g_lastDropsL1 = g_dropsCoreToRecvEdge;
  g_lastDropsL2 = g_dropsRecvEdgeToHost;

  g_tputFile << now << "," << total;
  for (uint32_t i = 0; i < NUM_SENDERS; ++i)
    {
      g_tputFile << "," << flowTput[i];
    }
  g_tputFile << "," << d1 << "," << d2 << "\n";

  g_cwndFile << now;
  for (uint32_t i = 0; i < NUM_SENDERS; ++i)
    {
      g_cwndFile << "," << g_flow[i].cwnd;
    }
  g_cwndFile << "\n";

  g_rttFile << now;
  for (uint32_t i = 0; i < NUM_SENDERS; ++i)
    {
      g_rttFile << "," << g_flow[i].rttMs;
    }
  g_rttFile << "\n";

  Simulator::Schedule (Seconds (0.1), &CalcThroughput);
}

static void
StartMeasurement ()
{
  for (uint32_t i = 0; i < NUM_SENDERS; ++i)
    {
      g_flow[i].lastTotalRx = g_sink[i]->GetTotalRx ();
    }
  Simulator::Schedule (Seconds (0.1), &CalcThroughput);
}

static void
ConnectSocketTraces ()
{
  for (uint32_t i = 0; i < NUM_SENDERS; ++i)
    {
      std::ostringstream cwndPath, rttPath, ifPath;
      cwndPath << "/NodeList/" << g_senderNodeId[i]
               << "/$ns3::TcpL4Protocol/SocketList/0/CongestionWindow";
      rttPath << "/NodeList/" << g_senderNodeId[i]
              << "/$ns3::TcpL4Protocol/SocketList/0/RTT";
      ifPath << "/NodeList/" << g_senderNodeId[i]
             << "/$ns3::TcpL4Protocol/SocketList/0/BytesInFlight";

      Config::ConnectWithoutContext (cwndPath.str (), MakeBoundCallback (&CwndTrace, i));
      Config::ConnectWithoutContext (rttPath.str (), MakeBoundCallback (&RttTrace, i));
      Config::ConnectWithoutContext (ifPath.str (), MakeBoundCallback (&InFlightTrace, i));
    }
}

static Ipv4InterfaceContainer
AssignLink (Ipv4AddressHelper &ipv4,
            NetDeviceContainer dev,
            uint32_t subnetId)
{
  std::ostringstream net;
  net << "10.1." << subnetId << ".0";
  ipv4.SetBase (net.str ().c_str (), "255.255.255.0");
  return ipv4.Assign (dev);
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
                          TypeIdValue (TcpFatTreeDrnn::GetTypeId ()));
    }
  else
    {
      NS_FATAL_ERROR ("Unknown transport: " << transport);
    }

  Config::SetDefault ("ns3::TcpSocket::SegmentSize", UintegerValue (1448));
  Config::SetDefault ("ns3::DropTailQueue<Packet>::MaxSize",
                      QueueSizeValue (QueueSize ("1p")));

  NodeContainer senders;
  senders.Create (NUM_SENDERS);
  NodeContainer edges;
  edges.Create (NUM_EDGES);
  NodeContainer aggs;
  aggs.Create (NUM_AGGS);
  NodeContainer core;
  core.Create (1);
  NodeContainer recvEdge;
  recvEdge.Create (1);
  NodeContainer receiver;
  receiver.Create (1);

  for (uint32_t i = 0; i < NUM_SENDERS; ++i)
    {
      g_senderNodeId[i] = senders.Get (i)->GetId ();
    }

  PointToPointHelper p2pSrv;
  p2pSrv.SetDeviceAttribute ("DataRate", StringValue ("1Gbps"));
  p2pSrv.SetChannelAttribute ("Delay", StringValue ("5us"));

  PointToPointHelper p2pFabric;
  p2pFabric.SetDeviceAttribute ("DataRate", StringValue ("1Gbps"));
  p2pFabric.SetChannelAttribute ("Delay", StringValue ("20us"));

  PointToPointHelper p2pUp;
  p2pUp.SetDeviceAttribute ("DataRate", StringValue ("300Mbps"));
  p2pUp.SetChannelAttribute ("Delay", StringValue ("100us"));

  PointToPointHelper p2pFinal;
  p2pFinal.SetDeviceAttribute ("DataRate", StringValue ("100Mbps"));
  p2pFinal.SetChannelAttribute ("Delay", StringValue ("50us"));

  std::vector<NetDeviceContainer> senderEdgeLinks;
  senderEdgeLinks.reserve (NUM_SENDERS);

  for (uint32_t i = 0; i < NUM_SENDERS; ++i)
    {
      uint32_t e = i / SENDERS_PER_EDGE;
      senderEdgeLinks.push_back (p2pSrv.Install (senders.Get (i), edges.Get (e)));
    }

  std::vector<NetDeviceContainer> edgeAggLinks;
  edgeAggLinks.reserve (NUM_EDGES * NUM_AGGS);
  for (uint32_t e = 0; e < NUM_EDGES; ++e)
    {
      for (uint32_t a = 0; a < NUM_AGGS; ++a)
        {
          edgeAggLinks.push_back (p2pFabric.Install (edges.Get (e), aggs.Get (a)));
        }
    }

  std::vector<NetDeviceContainer> aggCoreLinks;
  aggCoreLinks.reserve (NUM_AGGS);
  for (uint32_t a = 0; a < NUM_AGGS; ++a)
    {
      aggCoreLinks.push_back (p2pFabric.Install (aggs.Get (a), core.Get (0)));
    }

  NetDeviceContainer coreRecvEdge = p2pUp.Install (core.Get (0), recvEdge.Get (0));
  NetDeviceContainer recvEdgeHost = p2pFinal.Install (recvEdge.Get (0), receiver.Get (0));

  InternetStackHelper internet;
  internet.Install (senders);
  internet.Install (edges);
  internet.Install (aggs);
  internet.Install (core);
  internet.Install (recvEdge);
  internet.Install (receiver);

  TrafficControlHelper tchL1;
  tchL1.SetRootQueueDisc ("ns3::FifoQueueDisc", "MaxSize", StringValue ("40p"));
  QueueDiscContainer qdL1 = tchL1.Install (coreRecvEdge.Get (0));
  qdL1.Get (0)->TraceConnectWithoutContext ("Drop", MakeCallback (&DropTraceL1));

  TrafficControlHelper tchL2;
  tchL2.SetRootQueueDisc ("ns3::FifoQueueDisc", "MaxSize", StringValue ("20p"));
  QueueDiscContainer qdL2 = tchL2.Install (recvEdgeHost.Get (0));
  qdL2.Get (0)->TraceConnectWithoutContext ("Drop", MakeCallback (&DropTraceL2));

  Ipv4AddressHelper ipv4;
  uint32_t subnetId = 1;

  for (auto &d : senderEdgeLinks)
    {
      AssignLink (ipv4, d, subnetId++);
    }
  for (auto &d : edgeAggLinks)
    {
      AssignLink (ipv4, d, subnetId++);
    }
  for (auto &d : aggCoreLinks)
    {
      AssignLink (ipv4, d, subnetId++);
    }

  AssignLink (ipv4, coreRecvEdge, subnetId++);
  Ipv4InterfaceContainer recvIf = AssignLink (ipv4, recvEdgeHost, subnetId++);

  Ipv4GlobalRoutingHelper::PopulateRoutingTables ();
  Ipv4Address receiverAddr = recvIf.GetAddress (1);

  uint16_t basePort = 10000;
  for (uint32_t i = 0; i < NUM_SENDERS; ++i)
    {
      uint16_t port = basePort + i;

      PacketSinkHelper sinkHelper ("ns3::TcpSocketFactory",
                                   InetSocketAddress (Ipv4Address::GetAny (), port));
      ApplicationContainer sinkApp = sinkHelper.Install (receiver.Get (0));
      sinkApp.Start (Seconds (0.0));
      sinkApp.Stop (Seconds (g_simTime));
      g_sink[i] = DynamicCast<PacketSink> (sinkApp.Get (0));

      BulkSendHelper srcHelper ("ns3::TcpSocketFactory",
                                InetSocketAddress (receiverAddr, port));
      srcHelper.SetAttribute ("MaxBytes", UintegerValue (0));
      srcHelper.SetAttribute ("SendSize", UintegerValue (1448));
      ApplicationContainer srcApp = srcHelper.Install (senders.Get (i));

      // Near-synchronous start for incast micro-burst pressure
      double start = 2.0 + 0.001 * (i % SENDERS_PER_EDGE);
      srcApp.Start (Seconds (start));
      srcApp.Stop (Seconds (g_simTime));
    }

  std::string prefix = "ft_" + transport + "_";

  g_tputFile.open (prefix + "throughput.csv");
  g_tputFile << "Time,Total";
  for (uint32_t i = 0; i < NUM_SENDERS; ++i)
    {
      g_tputFile << ",Flow" << i;
    }
  g_tputFile << ",Drops_L1,Drops_L2\n";

  g_cwndFile.open (prefix + "cwnd.csv");
  g_cwndFile << "Time";
  for (uint32_t i = 0; i < NUM_SENDERS; ++i)
    {
      g_cwndFile << ",Flow" << i << "_CWND";
    }
  g_cwndFile << "\n";

  g_rttFile.open (prefix + "rtt.csv");
  g_rttFile << "Time";
  for (uint32_t i = 0; i < NUM_SENDERS; ++i)
    {
      g_rttFile << ",Flow" << i << "_RTT";
    }
  g_rttFile << "\n";

  Simulator::Schedule (Seconds (2.8), &ConnectSocketTraces);
  Simulator::Schedule (Seconds (3.0), &StartMeasurement);

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
      Simulator::Schedule (Seconds (3.2), &DrnnStep, 0.1);
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

  std::cout << "\n=== Fat-Tree Incast Simulation Complete ===\n"
            << "Transport:   " << transport << "\n"
            << "Duration:    " << g_simTime << " s\n"
            << "Senders:     " << NUM_SENDERS << "\n"
            << "Drops L1:    " << g_dropsCoreToRecvEdge << "\n"
            << "Drops L2:    " << g_dropsRecvEdgeToHost << "\n"
            << "CSV prefix:  " << prefix << "\n";

  return 0;
}

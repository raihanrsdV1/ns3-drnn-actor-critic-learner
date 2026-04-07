/*
 * lfn-satellite-sim.cc — Long Fat Network (LFN) / satellite-like link
 *
 * Scenario:
 *   Single TCP flow over a very high-BDP path.
 *
 * Topology:
 *   Sender --(10Gbps,1ms)-- R1 --(1Gbps,125ms)-- R2 --(10Gbps,1ms)-- Receiver
 *
 * The middle link gives ~250ms RTT baseline and huge BDP:
 *   BDP ~= 1Gbps * 0.25s = 31.25MB
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

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("LfnSatelliteSim");

class TcpLfnDrnn : public TcpCongestionOps
{
public:
  static TypeId GetTypeId (void);
  TcpLfnDrnn ();
  TcpLfnDrnn (const TcpLfnDrnn &sock);
  ~TcpLfnDrnn () override;

  std::string GetName () const override { return "TcpLfnDrnn"; }

  uint32_t GetSsThresh (Ptr<const TcpSocketState> tcb,
                        uint32_t bytesInFlight) override;
  void IncreaseWindow (Ptr<TcpSocketState> tcb,
                       uint32_t segmentsAcked) override;
  Ptr<TcpCongestionOps> Fork () override;

  static float g_target_cwnd;
};

static const uint32_t DRNN_MIN_CWND = 2 * 1448;
static const uint32_t DRNN_MAX_CWND = 40 * 1024 * 1024; // 40 MB
float TcpLfnDrnn::g_target_cwnd = static_cast<float> (256 * 1024); // 256 KB start

NS_OBJECT_ENSURE_REGISTERED (TcpLfnDrnn);

TypeId
TcpLfnDrnn::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::TcpLfnDrnn")
    .SetParent<TcpCongestionOps> ()
    .SetGroupName ("Internet")
    .AddConstructor<TcpLfnDrnn> ();
  return tid;
}

TcpLfnDrnn::TcpLfnDrnn () : TcpCongestionOps () {}
TcpLfnDrnn::TcpLfnDrnn (const TcpLfnDrnn &s) : TcpCongestionOps (s) {}
TcpLfnDrnn::~TcpLfnDrnn () {}

uint32_t
TcpLfnDrnn::GetSsThresh (Ptr<const TcpSocketState> tcb,
                         uint32_t /*bytesInFlight*/)
{
  return std::max<uint32_t> (DRNN_MIN_CWND, tcb->m_cWnd.Get () / 2);
}

void
TcpLfnDrnn::IncreaseWindow (Ptr<TcpSocketState> tcb, uint32_t segmentsAcked)
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
TcpLfnDrnn::Fork ()
{
  return CopyObject<TcpLfnDrnn> (this);
}

static const uint32_t NUM_FLOWS = 1;

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
static double g_simTime = 80.0;

static Ptr<OpenGymInterface> g_openGym;
static uint32_t g_gymPort = 5557;
static const uint32_t OBS_DIM = NUM_FLOWS * 3 + 2; // 5

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
      TcpLfnDrnn::g_target_cwnd = std::max ((float) DRNN_MIN_CWND,
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
                          TypeIdValue (TcpLfnDrnn::GetTypeId ()));
    }
  else
    {
      NS_FATAL_ERROR ("Unknown transport: " << transport);
    }

  Config::SetDefault ("ns3::TcpSocket::SegmentSize", UintegerValue (1448));
  Config::SetDefault ("ns3::TcpSocket::SndBufSize", UintegerValue (4194304)); // 4 MB
  Config::SetDefault ("ns3::TcpSocket::RcvBufSize", UintegerValue (6291456)); // 6 MB
  Config::SetDefault ("ns3::TcpSocketBase::WindowScaling", BooleanValue (true));
  Config::SetDefault ("ns3::DropTailQueue<Packet>::MaxSize",
                      QueueSizeValue (QueueSize ("1p")));

  NodeContainer sender;
  sender.Create (1);
  NodeContainer r1;
  r1.Create (1);
  NodeContainer r2;
  r2.Create (1);
  NodeContainer receiver;
  receiver.Create (1);

  g_srcNodeId[0] = sender.Get (0)->GetId ();

  PointToPointHelper p2pEdge;
  p2pEdge.SetDeviceAttribute ("DataRate", StringValue ("10Gbps"));
  p2pEdge.SetChannelAttribute ("Delay", StringValue ("1ms"));

  PointToPointHelper p2pLfn;
  p2pLfn.SetDeviceAttribute ("DataRate", StringValue ("1Gbps"));
  p2pLfn.SetChannelAttribute ("Delay", StringValue ("125ms"));

  NetDeviceContainer sendR1 = p2pEdge.Install (sender.Get (0), r1.Get (0));
  NetDeviceContainer r1R2 = p2pLfn.Install (r1.Get (0), r2.Get (0));
  NetDeviceContainer r2Recv = p2pEdge.Install (r2.Get (0), receiver.Get (0));

  InternetStackHelper internet;
  internet.Install (sender);
  internet.Install (r1);
  internet.Install (r2);
  internet.Install (receiver);

  // Queue 1: on LFN forward path
  TrafficControlHelper tch1;
  tch1.SetRootQueueDisc ("ns3::FifoQueueDisc", "MaxSize", StringValue ("3000p"));
  QueueDiscContainer qd1 = tch1.Install (r1R2.Get (0));
  qd1.Get (0)->TraceConnectWithoutContext ("Drop", MakeCallback (&DropTraceL1));

  // Queue 2: near receiver path (moderate queue)
  TrafficControlHelper tch2;
  tch2.SetRootQueueDisc ("ns3::FifoQueueDisc", "MaxSize", StringValue ("1000p"));
  QueueDiscContainer qd2 = tch2.Install (r2Recv.Get (0));
  qd2.Get (0)->TraceConnectWithoutContext ("Drop", MakeCallback (&DropTraceL2));

  Ipv4AddressHelper ipv4;
  ipv4.SetBase ("10.1.1.0", "255.255.255.0");
  ipv4.Assign (sendR1);

  ipv4.SetBase ("10.1.2.0", "255.255.255.0");
  ipv4.Assign (r1R2);

  ipv4.SetBase ("10.1.3.0", "255.255.255.0");
  Ipv4InterfaceContainer recvIf = ipv4.Assign (r2Recv);

  Ipv4GlobalRoutingHelper::PopulateRoutingTables ();

  uint16_t port = 9000;
  PacketSinkHelper sinkHelper ("ns3::TcpSocketFactory",
                               InetSocketAddress (Ipv4Address::GetAny (), port));
  ApplicationContainer sinkApp = sinkHelper.Install (receiver.Get (0));
  sinkApp.Start (Seconds (0.0));
  sinkApp.Stop (Seconds (g_simTime));
  g_sink[0] = DynamicCast<PacketSink> (sinkApp.Get (0));

  BulkSendHelper srcHelper ("ns3::TcpSocketFactory",
                            InetSocketAddress (recvIf.GetAddress (1), port));
  srcHelper.SetAttribute ("MaxBytes", UintegerValue (0));
  srcHelper.SetAttribute ("SendSize", UintegerValue (1448));
  ApplicationContainer srcApp = srcHelper.Install (sender.Get (0));
  srcApp.Start (Seconds (0.5));
  srcApp.Stop (Seconds (g_simTime));

  std::string prefix = "lf_" + transport + "_";

  g_tputFile.open (prefix + "throughput.csv");
  g_tputFile << "Time,Total,Flow0,Drops_L1,Drops_L2\n";

  g_cwndFile.open (prefix + "cwnd.csv");
  g_cwndFile << "Time,Flow0_CWND\n";

  g_rttFile.open (prefix + "rtt.csv");
  g_rttFile << "Time,Flow0_RTT\n";

  Simulator::Schedule (Seconds (1.5), &ConnectSocketTraces);
  Simulator::Schedule (Seconds (1.8), &StartMeasurement);

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
      Simulator::Schedule (Seconds (2.0), &DrnnStep, 0.1);
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

  std::cout << "\n=== LFN/Satellite Simulation Complete ===\n"
            << "Transport:   " << transport << "\n"
            << "Duration:    " << g_simTime << " s\n"
            << "RTT base:    ~250 ms\n"
            << "Bottleneck:  1 Gbps\n"
            << "Drops L1:    " << g_dropsL1 << "\n"
            << "Drops L2:    " << g_dropsL2 << "\n"
            << "CSV prefix:  " << prefix << "\n";

  return 0;
}

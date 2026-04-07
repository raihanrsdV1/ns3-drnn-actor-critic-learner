/*
 * Paper Replication Simulation
 * Based on: "A Deep RL-Based TCP Congestion Control Algorithm" (arXiv:2508.01047v3)
 *
 * EXACT paper spec (Section 3.3):
 *   - Single flow, dumbbell topology
 *   - Access links: 10 Mbps, bottleneck: 2 Mbps, propagation delay: 20ms
 *   - 5-dim state: [BytesInFlight, cWnd, RTT, SegmentsAcked, ssThresh]
 *   - 4 discrete actions (absolute byte deltas to cWnd):
 *       0: Maintain       (delta =    0)
 *       1: Standard Inc   (delta = +1500)
 *       2: Conservative Dec (delta = -150)
 *       3: Rocket Inc     (delta = +4000)
 *   - Reward = alpha*Throughput_Mbps - beta*RTT_sec  (beta=0.5 per paper)
 *   - Agent also sets ssThresh = new_cWnd (paper sets both simultaneously)
 *
 * Usage:
 *   ./ns3 run paper-sim -- --transport=cubic
 *   ./ns3 run paper-sim -- --transport=reno
 *   python3 scratch/paper-agent.py &
 *   ./ns3 run paper-sim -- --transport=drl
 */

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/traffic-control-module.h"
#include "ns3/applications-module.h"
#include "ns3/opengym-module.h"
#include <iostream>
#include <fstream>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("PaperSimulation");

// ================================================================
// CUSTOM CONGESTION CONTROL: TcpDrl
// Paper-faithful agent-controlled TCP.
// The Python DQN agent picks an action every 100ms.
// IncreaseWindow applies that action on every ACK as a linear
// multiplier on the standard AIMD base rate.
// ================================================================
class TcpDrl : public TcpCongestionOps
{
public:
    static TypeId GetTypeId (void);
    TcpDrl ();
    TcpDrl (const TcpDrl &sock);
    ~TcpDrl () override;

    std::string GetName () const override;
    uint32_t GetSsThresh (Ptr<const TcpSocketState> tcb, uint32_t bytesInFlight) override;
    void IncreaseWindow (Ptr<TcpSocketState> tcb, uint32_t segmentsAcked) override;
    Ptr<TcpCongestionOps> Fork () override;

    // Action set by the Python agent each step via OpenGym.
    // Paper Section 3.3 — absolute byte deltas applied directly to cWnd:
    //   0: Maintain        (delta =    0) — hold steady
    //   1: Standard Inc    (delta = +1500) — steady growth
    //   2: Conservative Dec(delta =  -150) — gentle pullback
    //   3: Rocket Inc      (delta = +4000) — fast startup / recovery
    static uint32_t g_action;
    static int32_t  g_cwndDelta; // resolved delta for current action
};

uint32_t TcpDrl::g_action    = 1; // Default: Standard Increase
int32_t  TcpDrl::g_cwndDelta = 1500;

// BDP = 2 Mbps × (20+1+1)ms ≈ 5,500 bytes. Queue=100p ≈ 150 KB headroom.
// No hard MAX_CWND — let the queue drops do the limiting, same as paper.

NS_OBJECT_ENSURE_REGISTERED (TcpDrl);

TypeId
TcpDrl::GetTypeId (void)
{
    static TypeId tid = TypeId ("ns3::TcpDrl")
        .SetParent<TcpCongestionOps> ()
        .SetGroupName ("Internet")
        .AddConstructor<TcpDrl> ();
    return tid;
}

TcpDrl::TcpDrl () : TcpCongestionOps () {}
TcpDrl::TcpDrl (const TcpDrl &sock) : TcpCongestionOps (sock) {}
TcpDrl::~TcpDrl () {}

std::string TcpDrl::GetName () const { return "TcpDrl"; }

uint32_t
TcpDrl::GetSsThresh (Ptr<const TcpSocketState> tcb, uint32_t bytesInFlight)
{
    // On loss, reduce cwnd by one conservative-decrease step (-150 bytes)
    // rather than the standard /2. The agent's reward already penalises high
    // RTT (which precedes loss), so it learns to back off before drops happen.
    // Halving would crater the window and make recovery take many steps,
    // preventing the agent from learning the throughput-RTT tradeoff clearly.
    uint32_t reduced = (tcb->m_cWnd.Get () > 150)
                       ? tcb->m_cWnd.Get () - 150
                       : 2 * tcb->m_segmentSize;
    return std::max<uint32_t> (2 * tcb->m_segmentSize, reduced);
}

// ================================================================
// GLOBAL STATE — shared between TcpDrl and OpenGym callbacks
// ================================================================
static uint32_t g_cwnd = 0;
static uint32_t g_rtt = 0;               // milliseconds
static uint32_t g_bytesInFlight = 0;
static uint32_t g_segmentsAcked = 0;     // accumulated per OpenGym step
static uint32_t g_ssThresh = 65535;      // slow start threshold
static uint32_t g_packetDrops = 0;       // drops per OpenGym step
static uint32_t g_totalDrops = 0;        // total drops over entire simulation
static uint32_t g_intervalDrops = 0;     // drops in current 0.1s throughput window

void
TcpDrl::IncreaseWindow (Ptr<TcpSocketState> tcb, uint32_t segmentsAcked)
{
    if (segmentsAcked == 0)
        return;

    // Track segments acked for the state vector
    g_segmentsAcked += segmentsAcked;

    // Paper Section 3.3: the action IS the delta added directly to cWnd.
    // action_mapping = {0: 0, 1: +1500, 2: -150, 3: +4000}
    // Applied once per OpenGym step (every 0.1s), NOT per ACK.
    // We only apply it on the FIRST ACK of the step to avoid repeated application.
    if (segmentsAcked > 0 && g_cwndDelta != 0)
    {
        int32_t newCwnd = (int32_t)tcb->m_cWnd.Get () + g_cwndDelta;
        uint32_t minCwnd = 2 * tcb->m_segmentSize;
        tcb->m_cWnd = (uint32_t)std::max<int32_t> ((int32_t)minCwnd, newCwnd);
        // Paper also sets ssThresh = new_cWnd
        tcb->m_ssThresh = tcb->m_cWnd.Get ();
        g_ssThresh = tcb->m_ssThresh;
        g_cwndDelta = 0; // consumed — reset until next agent step
    }
}

Ptr<TcpCongestionOps>
TcpDrl::Fork ()
{
    return CopyObject<TcpDrl> (this);
}

// ================================================================
// DATA COLLECTION
// ================================================================
Ptr<OpenGymInterface> openGymInterface;
std::ofstream tpFile, rttFile, cwndFile, dropFile;
uint32_t g_bytesRx = 0;
uint32_t g_lastBytesRx = 0;

// --- Trace callbacks ---
void CwndTrace (uint32_t oldVal, uint32_t newVal)
{
    g_cwnd = newVal;
    cwndFile << Simulator::Now ().GetSeconds () << ",0," << newVal << std::endl;
}

void SsThreshTrace (uint32_t oldVal, uint32_t newVal)
{
    g_ssThresh = newVal;
}

void RttTrace (Time oldVal, Time newVal)
{
    g_rtt = newVal.GetMilliSeconds ();
    rttFile << Simulator::Now ().GetSeconds () << ",0," << newVal.GetMilliSeconds () << std::endl;
}

void InFlightTrace (uint32_t oldVal, uint32_t newVal)
{
    g_bytesInFlight = newVal;
}

void DropTrace (Ptr<const QueueDiscItem> item)
{
    g_intervalDrops++;
    g_packetDrops++;
    g_totalDrops++;
}

void RxTrace (Ptr<const Packet> p, const Address &addr)
{
    g_bytesRx += p->GetSize ();
}

void CalcThroughput ()
{
    double now = Simulator::Now ().GetSeconds ();
    uint32_t diff = g_bytesRx - g_lastBytesRx;
    double tput = (diff * 8.0) / (0.1 * 1e6); // Mbps over 0.1s window
    g_lastBytesRx = g_bytesRx;
    tpFile   << now << ",0," << tput             << std::endl;
    dropFile << now << ",0," << g_intervalDrops  << std::endl;
    // Also snapshot CWND and RTT every interval so DRL lines are continuous
    // even when the event-driven traces fire infrequently (e.g. MAINTAIN action).
    if (g_cwnd > 0)
        cwndFile << now << ",0," << g_cwnd << std::endl;
    if (g_rtt > 0)
        rttFile  << now << ",0," << g_rtt  << std::endl;
    g_intervalDrops = 0;  // reset for next 0.1s window
    Simulator::Schedule (Seconds (0.1), &CalcThroughput);
}

void ConnectTraces (uint32_t nodeId)
{
    std::string base = "/NodeList/" + std::to_string (nodeId) +
                       "/$ns3::TcpL4Protocol/SocketList/*/";
    Config::ConnectWithoutContext (base + "CongestionWindow", MakeCallback (&CwndTrace));
    Config::ConnectWithoutContext (base + "RTT",              MakeCallback (&RttTrace));
    Config::ConnectWithoutContext (base + "BytesInFlight",    MakeCallback (&InFlightTrace));
    Config::ConnectWithoutContext (base + "SlowStartThreshold", MakeCallback (&SsThreshTrace));
    std::cout << "Traces connected for node " << nodeId << " at t="
              << Simulator::Now ().GetSeconds () << "s" << std::endl;
}

// ================================================================
// OPENGYM CALLBACKS (only used in DRL mode)
// ================================================================

// Paper's state space: [BytesInFlight, cWnd, RTT, SegmentsAcked, ssThresh]
Ptr<OpenGymSpace> GetObsSpace ()
{
    return CreateObject<OpenGymBoxSpace> (
        0.0f, 1e7f,
        std::vector<uint32_t>{5},
        TypeNameGet<float> ()
    );
}

// 4 discrete actions (paper Section 3.3)
Ptr<OpenGymSpace> GetActSpace ()
{
    return CreateObject<OpenGymDiscreteSpace> (4);
}

Ptr<OpenGymDataContainer> GetObs ()
{
    auto box = CreateObject<OpenGymBoxContainer<float>> (std::vector<uint32_t>{5});
    box->AddValue ((float)g_bytesInFlight);
    box->AddValue ((float)g_cwnd);
    box->AddValue ((float)g_rtt);
    box->AddValue ((float)g_segmentsAcked);
    box->AddValue ((float)g_ssThresh);
    return box;
}

float GetReward ()
{
    // Paper Section 3.3: Reward = alpha*Throughput - beta*Latency
    // Where Throughput is in Mbps and Latency is RTT in seconds.
    // beta was tuned to 0.5 (paper: "double_pen = -0.5").
    if (g_rtt == 0 || g_cwnd == 0)
        return 0.0f;

    // Throughput estimate: cwnd / RTT  (bytes/sec -> Mbps)
    float rttSec     = (float)g_rtt / 1000.0f;
    float throughput = ((float)g_cwnd / rttSec) / 1e6f;  // Mbps

    // Latency penalty (RTT in seconds)
    float latency = rttSec;

    return throughput - 0.5f * latency;
}

bool ExecuteAction (Ptr<OpenGymDataContainer> action)
{
    auto disc = DynamicCast<OpenGymDiscreteContainer> (action);
    if (disc) {
        TcpDrl::g_action = disc->GetValue ();
        // Paper action_mapping (Section 3.3):
        //   0 -> 0     Maintain
        //   1 -> +1500 Standard Increase
        //   2 -> -150  Conservative Decrease
        //   3 -> +4000 Rocket Increase
        static const int32_t DELTA[4] = {0, 1500, -150, 4000};
        uint32_t a = TcpDrl::g_action;
        TcpDrl::g_cwndDelta = (a < 4) ? DELTA[a] : 0;
    }
    // Reset per-step counters
    g_segmentsAcked = 0;
    g_packetDrops   = 0;
    return true;
}

bool GameOver () { return false; }

std::string ExtraInfo ()
{
    // Pass total drops to agent for logging
    return std::to_string (g_totalDrops);
}

void StepSchedule (double dt, Ptr<OpenGymInterface> iface)
{
    Simulator::Schedule (Seconds (dt), &StepSchedule, dt, iface);
    iface->NotifyCurrentState ();
}

// ================================================================
// MAIN
// ================================================================
int main (int argc, char *argv[])
{
    std::string transport = "drl";
    std::string name = "";
    double simTime = 30.0;   // Extended to 30 seconds for more sawtooth cycles
    uint16_t port = 5556;

    // Standard Ethernet MSS: 1500 MTU - 20 IP - 20 TCP - 12 TCP options = 1448.
    // ns-3 default is only 536 bytes — much too small, changes queue dynamics.
    Config::SetDefault ("ns3::TcpSocket::SegmentSize",  UintegerValue (1448));
    // TCP buffer must be larger than BDP+queue (~156 KB) so Cubic/Reno can fill
    // the 100-packet bottleneck queue and trigger real drops.
    Config::SetDefault ("ns3::TcpSocket::SndBufSize", UintegerValue (1 << 20)); // 1 MB
    Config::SetDefault ("ns3::TcpSocket::RcvBufSize", UintegerValue (1 << 20)); // 1 MB
    // Keep SACK enabled (ns-3 default). With SACK, fast retransmit fires on
    // each drop burst, cwnd is halved, and Cubic probes back to W_max quickly
    // — giving a cycle every ~1-2 seconds = the sawtooth pattern in the paper.
    // Without SACK, full RTO fires (sender silenced 200-800ms + slow-start),
    // making each cycle 4-6 seconds — only ONE cycle visible in 10 seconds.
    //
    // Set InitialSlowStartThreshold very large so TCP stays in exponential
    // slow-start until the FIRST actual packet loss (not the ns-3 default of
    // 65535 bytes which exits slow-start at 65 KB — well before the drop
    // threshold of ~156 KB — and then Cubic CA grows slowly taking ~5 seconds
    // per cycle, giving only ONE sawtooth in 10s instead of 4-5).
    // With large ssThresh: slow-start reaches 156 KB in ~4 RTTs (~180ms),
    // drops fire, Cubic recovers in ~1-2s, giving 4-5 cycles in 10s.
    Config::SetDefault ("ns3::TcpSocket::InitialSlowStartThreshold",
                        UintegerValue (10 * 1024 * 1024)); // 10 MB ≈ infinite

    CommandLine cmd;
    cmd.AddValue ("transport", "Congestion control: drl, cubic, or reno", transport);
    cmd.AddValue ("name", "Output file prefix (default: p_<transport>)", name);
    cmd.AddValue ("simTime", "Simulation duration in seconds", simTime);
    cmd.AddValue ("port", "OpenGym port (DRL mode only)", port);
    cmd.Parse (argc, argv);

    if (name.empty ())
        name = "p_" + transport;

    bool useDrl = (transport == "drl");

    // --- Set congestion control algorithm ---
    if (useDrl) {
        Config::SetDefault ("ns3::TcpL4Protocol::SocketType",
                            TypeIdValue (TcpDrl::GetTypeId ()));
        std::cout << "Congestion control: TcpDrl (agent-controlled)" << std::endl;
    } else if (transport == "cubic") {
        Config::SetDefault ("ns3::TcpL4Protocol::SocketType",
                            TypeIdValue (TypeId::LookupByName ("ns3::TcpCubic")));
        std::cout << "Congestion control: TCP Cubic" << std::endl;
    } else {
        Config::SetDefault ("ns3::TcpL4Protocol::SocketType",
                            TypeIdValue (TypeId::LookupByName ("ns3::TcpNewReno")));
        std::cout << "Congestion control: TCP NewReno" << std::endl;
    }

    // --- Open CSV output files ---
    tpFile.open   (name + "_throughput.csv");
    rttFile.open  (name + "_rtt.csv");
    cwndFile.open (name + "_cwnd.csv");
    dropFile.open (name + "_drops.csv");
    tpFile   << "Time,FlowID,Throughput_Mbps" << std::endl;
    rttFile  << "Time,FlowID,RTT_ms"          << std::endl;
    cwndFile << "Time,FlowID,Cwnd_Bytes"      << std::endl;
    dropFile << "Time,FlowID,Drops"           << std::endl;

    // ================================================================
    // TOPOLOGY: Dumbbell with single flow  (paper Section 3.2)
    //   Sender(0) --[10Mbps/1ms]--> GW1(1) --[2Mbps/20ms]--> GW2(2) --[10Mbps/1ms]--> Receiver(3)
    //   Min RTT = 2*(1+20+1) = 44 ms
    //   BDP    = 2 Mbps * 44 ms / 8 = 11,000 bytes ≈ 10.75 KB
    //   Queue  = 100 packets * ~1500 B = ~146 KB
    //   Drops start when cWnd > BDP + Queue ≈ 157 KB
    // ================================================================
    NodeContainer nodes;
    nodes.Create (4);

    PointToPointHelper access;
    access.SetDeviceAttribute ("DataRate", StringValue ("10Mbps"));  // Paper: 10 Mbps
    access.SetChannelAttribute ("Delay", StringValue ("1ms"));

    PointToPointHelper bottleneck;
    bottleneck.SetDeviceAttribute ("DataRate", StringValue ("2Mbps")); // Paper: exactly 2 Mbps
    bottleneck.SetChannelAttribute ("Delay", StringValue ("20ms"));
    // Device queue must be tiny (1p) — the FifoQueueDisc above is the real 100p queue.
    // Without this, we get DOUBLE buffering: 100p device + 100p QueueDisc = 200p,
    // which halves the effective congestion and explains why drops were so low.
    bottleneck.SetQueue ("ns3::DropTailQueue", "MaxSize", StringValue ("1p"));

    NetDeviceContainer senderDev   = access.Install (nodes.Get (0), nodes.Get (1));
    NetDeviceContainer bnDev       = bottleneck.Install (nodes.Get (1), nodes.Get (2));
    NetDeviceContainer receiverDev = access.Install (nodes.Get (2), nodes.Get (3));

    InternetStackHelper stack;
    stack.Install (nodes);

    // Install a 100-packet FifoQueueDisc on the bottleneck (paper: 100-packet DropTail).
    // In ns-3's architecture, the TC QueueDisc sits above the NetDevice and is
    // where congestion drops actually occur. "Drop" fires whenever the 100p limit
    // is exceeded — this is what Cubic triggers aggressively.
    TrafficControlHelper tch;
    tch.SetRootQueueDisc ("ns3::FifoQueueDisc", "MaxSize", StringValue ("100p"));
    QueueDiscContainer qdiscs = tch.Install (bnDev);
    // qdiscs.Get(0) = GW1 egress toward GW2 — the congested bottleneck direction
    qdiscs.Get (0)->TraceConnectWithoutContext ("Drop", MakeCallback (&DropTrace));

    // --- IP addressing ---
    Ipv4AddressHelper ipv4;
    ipv4.SetBase ("10.0.1.0", "255.255.255.0");
    ipv4.Assign (senderDev);
    ipv4.SetBase ("10.0.2.0", "255.255.255.0");
    ipv4.Assign (bnDev);
    ipv4.SetBase ("10.0.3.0", "255.255.255.0");
    Ipv4InterfaceContainer recvIface = ipv4.Assign (receiverDev);

    Ipv4GlobalRoutingHelper::PopulateRoutingTables ();

    // --- Traffic: single bulk TCP flow ---
    uint16_t sinkPort = 9000;
    Address sinkAddr (InetSocketAddress (recvIface.GetAddress (1), sinkPort));

    PacketSinkHelper sink ("ns3::TcpSocketFactory",
                           InetSocketAddress (Ipv4Address::GetAny (), sinkPort));
    ApplicationContainer sinkApp = sink.Install (nodes.Get (3));
    sinkApp.Start (Seconds (0.0));
    sinkApp.Stop (Seconds (simTime));
    sinkApp.Get (0)->TraceConnectWithoutContext ("Rx", MakeCallback (&RxTrace));

    BulkSendHelper source ("ns3::TcpSocketFactory", sinkAddr);
    source.SetAttribute ("MaxBytes", UintegerValue (0)); // unlimited
    ApplicationContainer srcApp = source.Install (nodes.Get (0));
    srcApp.Start (Seconds (1.0));
    srcApp.Stop (Seconds (simTime));

    // Connect TCP socket traces 0.1s after flow starts
    Simulator::Schedule (Seconds (1.1), &ConnectTraces, nodes.Get (0)->GetId ());

    // --- Throughput sampling ---
    Simulator::Schedule (Seconds (0.1), &CalcThroughput);

    // --- OpenGym (DRL mode only) ---
    if (useDrl) {
        openGymInterface = CreateObject<OpenGymInterface> (port);
        openGymInterface->SetGetActionSpaceCb      (MakeCallback (&GetActSpace));
        openGymInterface->SetGetObservationSpaceCb  (MakeCallback (&GetObsSpace));
        openGymInterface->SetGetObservationCb       (MakeCallback (&GetObs));
        openGymInterface->SetGetRewardCb            (MakeCallback (&GetReward));
        openGymInterface->SetGetGameOverCb          (MakeCallback (&GameOver));
        openGymInterface->SetGetExtraInfoCb         (MakeCallback (&ExtraInfo));
        openGymInterface->SetExecuteActionsCb       (MakeCallback (&ExecuteAction));

        std::cout << "\n=====================================================" << std::endl;
        std::cout << "  DRL SIMULATION — waiting for agent on port " << port << std::endl;
        std::cout << "  Run:  python3 scratch/paper-agent.py" << std::endl;
        std::cout << "=====================================================" << std::endl;

        // Start agent communication 0.5s after flow starts
        Simulator::Schedule (Seconds (1.5), &StepSchedule, 0.1, openGymInterface);
    } else {
        std::cout << "\n=====================================================" << std::endl;
        std::cout << "  BASELINE: " << transport << " (no agent needed)" << std::endl;
        std::cout << "=====================================================" << std::endl;
    }

    // --- Run ---
    Simulator::Stop(Seconds (simTime));
    Simulator::Run();

    if (useDrl)
        openGymInterface->NotifySimulationEnd ();

    tpFile.close ();
    rttFile.close ();
    cwndFile.close ();
    dropFile.close ();

    std::cout << "\n--- Simulation complete ---" << std::endl;
    std::cout << "Transport:     " << transport << std::endl;
    std::cout << "Duration:      " << simTime << "s" << std::endl;
    std::cout << "Packet drops:  " << g_totalDrops << std::endl;
    std::cout << "Output files:  " << name << "_throughput.csv, "
              << name << "_rtt.csv, " << name << "_cwnd.csv, "
              << name << "_drops.csv" << std::endl;

    Simulator::Destroy ();
    return 0;
}

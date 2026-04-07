/*
 * DRNN-TCP Simulation for NS-3.40
 * Features:
 * - Same topology as tcp-evolution (6 senders, dumbbell, 6 receivers)
 * - OpenGym interface for DRNN agent control
 * - Metrics: Throughput, RTT, Cwnd (saved to drnn_*.csv)
 */

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/applications-module.h"
#include "ns3/opengym-module.h"
#include <iostream>
#include <fstream>
#include <map>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("DrnnTcpSimulation");

// =============================================
// CUSTOM CONGESTION CONTROL: TcpDrnn
// Agent-controlled congestion window management.
// This replaces Cubic/NewReno — the DRNN agent
// decides how to adjust cwnd on every ACK.
// =============================================
class TcpDrnn : public TcpCongestionOps
{
public:
    static TypeId GetTypeId (void);
    TcpDrnn ();
    TcpDrnn (const TcpDrnn &sock);
    ~TcpDrnn () override;

    std::string GetName () const override;
    uint32_t GetSsThresh (Ptr<const TcpSocketState> tcb, uint32_t bytesInFlight) override;
    void IncreaseWindow (Ptr<TcpSocketState> tcb, uint32_t segmentsAcked) override;
    Ptr<TcpCongestionOps> Fork () override;

    // Static action variable set by the Python agent via OpenGym.
    // Actions are AIMD multipliers: 0=FastDec(-4x), 1=SlowDec(-1x), 2=Maintain,
    // 3=AIMD(+1x), 4=ModInc(+2x), 5=FastInc(+4x), 6=VeryFast(+8x)
    static uint32_t g_action;
};

uint32_t TcpDrnn::g_action = 3; // Default: standard AIMD (converges like TCP Reno)

// Safety cap: max cwnd per flow based on the network's Bandwidth-Delay Product.
// BDP = bottleneck_rate × min_RTT = 10 Mbps × 42ms = 52,500 bytes.
// We allow 3× BDP to tolerate some queuing, but no more.
// Any cwnd above BDP just fills the router queue for zero throughput gain.
static const uint32_t MAX_CWND_BYTES = 52500; // 1x BDP = 10 Mbps * 42ms

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

TcpDrnn::TcpDrnn () : TcpCongestionOps () {}
TcpDrnn::TcpDrnn (const TcpDrnn &sock) : TcpCongestionOps (sock) {}
TcpDrnn::~TcpDrnn () {}

std::string
TcpDrnn::GetName () const
{
    return "TcpDrnn";
}

uint32_t
TcpDrnn::GetSsThresh (Ptr<const TcpSocketState> tcb, uint32_t bytesInFlight)
{
    // Standard loss response: halve cwnd. The agent controls growth rate,
    // not loss response — this matches how real TCP operates.
    uint32_t minSsThresh = 2 * tcb->m_segmentSize;
    return std::max<uint32_t> (minSsThresh, tcb->m_cWnd.Get () / 2);
}

void
TcpDrnn::IncreaseWindow (Ptr<TcpSocketState> tcb, uint32_t segmentsAcked)
{
    // *** AGENT-CONTROLLED CONGESTION WINDOW ***
    // All 7 actions use the AIMD base rate (~ +1 MSS per RTT) with different
    // multipliers. This keeps ALL growth/decay LINEAR — no exponential blowup.
    // The paper's key insight: the DRL agent learns the optimal rate multiplier
    // for current network conditions through the reward function.

    if (segmentsAcked == 0)
        return;

    uint32_t segSize = tcb->m_segmentSize;
    uint32_t cwnd = tcb->m_cWnd.Get ();
    uint32_t minCwnd = 2 * segSize;

    // Base AIMD rate: segSize^2 / cwnd per ACK ~ +1 MSS per RTT (standard TCP CA)
    int32_t baseRate = std::max<int32_t> (1, (int32_t)((segSize * segSize) / cwnd));

    // Action -> AIMD multiplier (negative = decrease, 0 = hold, positive = increase)
    int32_t multiplier;
    switch (g_action) {
        case 0: multiplier = -4; break; // FAST_DEC:     drain queues quickly
        case 1: multiplier = -1; break; // SLOW_DEC:     gentle pullback
        case 2: multiplier =  0; break; // MAINTAIN:     hold current rate
        case 3: multiplier =  1; break; // AIMD:         standard TCP congestion avoidance
        case 4: multiplier =  2; break; // MODERATE_INC: 2x standard growth
        case 5: multiplier =  4; break; // FAST_INC:     4x standard (undershoot recovery)
        case 6: multiplier =  8; break; // VERY_FAST:    8x standard (startup ramp-up)
        default: multiplier = 1; break;
    }

    int32_t newCwnd = (int32_t)cwnd + baseRate * multiplier;

    // Clamp to [minCwnd, MAX_CWND_BYTES] — hard physical constraint
    tcb->m_cWnd = std::min<uint32_t> (
        MAX_CWND_BYTES,
        std::max<uint32_t> (minCwnd, (uint32_t)std::max<int32_t> (0, newCwnd))
    );
}

Ptr<TcpCongestionOps>
TcpDrnn::Fork ()
{
    return CopyObject<TcpDrnn> (this);
}

// --- GLOBAL VARIABLES ---
Ptr<OpenGymInterface> openGymInterface;

// --- DATA COLLECTION VARIABLES ---
std::ofstream throughputStream;
std::ofstream rttStream;
std::ofstream cwndStream;

std::map<uint32_t, uint32_t> bytesReceived;
std::map<uint32_t, uint32_t> lastBytesReceived;

// State tracking for AI
struct FlowState {
    uint32_t cWnd = 0;
    uint32_t rtt = 0;
    uint32_t bytesInFlight = 0;
};
std::map<uint32_t, FlowState> g_flowStates; // Per-flow state

// --- TRACE CALLBACKS ---
void CwndTracer (uint32_t flowId, uint32_t socketId, uint32_t oldVal, uint32_t newVal) {
    g_flowStates[flowId].cWnd = newVal;
    cwndStream << Simulator::Now().GetSeconds() << "," << flowId << "," << newVal << std::endl;
}

void RttTracer (uint32_t flowId, uint32_t socketId, Time oldVal, Time newVal) {
    g_flowStates[flowId].rtt = newVal.GetMilliSeconds();
    rttStream << Simulator::Now().GetSeconds() << "," << flowId << "," << newVal.GetMilliSeconds() << std::endl;
}

void InFlightTracer (uint32_t flowId, uint32_t oldVal, uint32_t newVal) {
    g_flowStates[flowId].bytesInFlight = newVal;
}

void RxDrop (Ptr<const Packet> p) {
    NS_LOG_UNCOND("Packet Loss at " << Simulator::Now().GetSeconds());
}

void ConnectSocketTraces (uint32_t nodeId, uint32_t flowId) {
    std::stringstream pathCwnd, pathRtt, pathInFlight;
    pathCwnd << "/NodeList/" << nodeId << "/$ns3::TcpL4Protocol/SocketList/*/CongestionWindow";
    pathRtt  << "/NodeList/" << nodeId << "/$ns3::TcpL4Protocol/SocketList/*/RTT";
    pathInFlight << "/NodeList/" << nodeId << "/$ns3::TcpL4Protocol/SocketList/*/BytesInFlight";

    Config::ConnectWithoutContext (pathCwnd.str(), MakeBoundCallback (&CwndTracer, flowId, (uint32_t)0));
    Config::ConnectWithoutContext (pathRtt.str(),  MakeBoundCallback (&RttTracer,  flowId, (uint32_t)0));
    Config::ConnectWithoutContext (pathInFlight.str(), MakeBoundCallback (&InFlightTracer, flowId));

    std::cout << "DEBUG: Traces attached for flow " << flowId << " (node " << nodeId << ") at " << Simulator::Now().GetSeconds() << "s" << std::endl;
}

void CalculateThroughput () {
    double now = Simulator::Now().GetSeconds();
    
    for (int i=0; i<6; i++) {
        uint32_t currentBytes = bytesReceived[i];
        uint32_t diff = currentBytes - lastBytesReceived[i];
        double throughput = (diff * 8.0) / (0.1 * 1000000.0);
        lastBytesReceived[i] = currentBytes;
        throughputStream << now << "," << i << "," << throughput << std::endl;
    }
    
    Simulator::Schedule (Seconds (0.1), &CalculateThroughput);
}

void PacketRx (uint32_t flowId, Ptr<const Packet> p, const Address &addr) {
    bytesReceived[flowId] += p->GetSize();
}

// --- OPENGYM INTERFACE ---
Ptr<OpenGymSpace> GetObservationSpace() {
    float low = 0.0;
    float high = 10000000.0;
    std::vector<uint32_t> shape = {18,}; // 6 flows × 3 metrics (cwnd, rtt, inflight)
    std::string dtype = TypeNameGet<float> ();
    return CreateObject<OpenGymBoxSpace> (low, high, shape, dtype);
}

Ptr<OpenGymSpace> GetActionSpace() {
    return CreateObject<OpenGymDiscreteSpace> (7); // 7 actions: LargeDec/SmallDec/Maintain/SmallInc/MedInc/LargeInc/Aggressive
}

Ptr<OpenGymDataContainer> GetObservation() {
    Ptr<OpenGymBoxContainer<float>> box = CreateObject<OpenGymBoxContainer<float>> (std::vector<uint32_t> {18});
    
    // Pack all 6 flows: [flow0_cwnd, flow0_rtt, flow0_inflight, flow1_cwnd, ...]
    for (int i=0; i<6; i++) {
        box->AddValue((float)g_flowStates[i].cWnd);
        box->AddValue((float)g_flowStates[i].rtt);
        box->AddValue((float)g_flowStates[i].bytesInFlight);
    }
    return box;
}

float GetReward() {
    // Paper-aligned reward: throughput utilization - heavy latency penalty + fairness.
    // The paper's DRL achieved 46% lower RTT by heavily penalizing queue buildup.
    float totalTputBps = 0.0;
    float avgRtt = 0.0;
    int activeFlows = 0;
    
    for (int i=0; i<6; i++) {
        if (g_flowStates[i].rtt > 0 && g_flowStates[i].cWnd > 0) {
            float rttSec = (float)g_flowStates[i].rtt / 1000.0f;
            totalTputBps += (float)g_flowStates[i].cWnd / rttSec;
            avgRtt += g_flowStates[i].rtt;
            activeFlows++;
        }
    }
    
    if (activeFlows == 0) return 0.0f;
    avgRtt /= activeFlows;
    
    // 1. Throughput utilization: capped at 1.0 (no bonus for overshooting the pipe)
    float utilization = std::min(totalTputBps / 1250000.0f, 1.0f);
    
    // 2. Latency penalty — STRONG (this is the paper's key advantage over Cubic)
    //    min RTT ~42ms. Any increase = queue buildup = wasted buffer.
    //    At 2x min RTT (84ms), penalty = -0.8, wiping out throughput reward.
    float rttRatio = avgRtt / 42.0f;
    float rttPenalty = 0.0f;
    if (rttRatio > 1.1f) {
        rttPenalty = -0.8f * (rttRatio - 1.0f);
    }
    
    // 3. Fairness bonus (Jain's index: 1.0 = perfectly fair)
    float fairnessBonus = 0.0f;
    if (activeFlows >= 2) {
        float sumTput = 0, sumTputSq = 0;
        for (int i=0; i<6; i++) {
            if (g_flowStates[i].cWnd > 0 && g_flowStates[i].rtt > 0) {
                float rttSec = (float)g_flowStates[i].rtt / 1000.0f;
                float tput = (float)g_flowStates[i].cWnd / rttSec;
                sumTput += tput;
                sumTputSq += tput * tput;
            }
        }
        if (sumTputSq > 0) {
            float jain = (sumTput * sumTput) / (activeFlows * sumTputSq);
            fairnessBonus = 0.3f * (jain - 1.0f);
        }
    }
    
    return utilization + rttPenalty + fairnessBonus;
}

bool ExecuteAction(Ptr<OpenGymDataContainer> action) {
    // Parse the agent's action and apply it to TcpDrnn congestion control
    Ptr<OpenGymDiscreteContainer> discrete = DynamicCast<OpenGymDiscreteContainer>(action);
    if (discrete) {
        uint32_t actionValue = discrete->GetValue();
        TcpDrnn::g_action = actionValue;

        static const char* actionNames[] = {"FAST_DEC", "SLOW_DEC", "MAINTAIN", "AIMD", "MODERATE", "FAST_INC", "VERY_FAST"};
        if (actionValue < 7) {
            NS_LOG_INFO("Agent action: " << actionNames[actionValue]
                        << " (" << actionValue << ") at t=" << Simulator::Now().GetSeconds() << "s");
        }
    }
    return true;
}

bool GetGameOver() { return false; }
std::string GetExtraInfo() { return ""; }

void ScheduleNextStateRead(double envStepTime, Ptr<OpenGymInterface> openGym) {
    Simulator::Schedule(Seconds(envStepTime), &ScheduleNextStateRead, envStepTime, openGym);
    openGym->NotifyCurrentState();
}

int main (int argc, char *argv[]) {
    LogComponentEnable ("DrnnTcpSimulation", LOG_LEVEL_INFO);
    LogComponentEnable ("OpenGymInterface", LOG_LEVEL_INFO);

    uint16_t openGymPort = 5555;
    CommandLine cmd;
    cmd.AddValue ("openGymPort", "Port number for OpenGym", openGymPort);
    cmd.Parse (argc, argv);

    // *** USE OUR CUSTOM DRNN CONGESTION CONTROL (NOT Cubic, NOT NewReno) ***
    Config::SetDefault ("ns3::TcpL4Protocol::SocketType", TypeIdValue (TcpDrnn::GetTypeId ()));
    NS_LOG_INFO ("Congestion control set to: TcpDrnn (agent-controlled)");

    // Open Data Files
    throughputStream.open ("drnn_throughput.csv");
    rttStream.open ("drnn_rtt.csv");
    cwndStream.open ("drnn_cwnd.csv");
    
    throughputStream << "Time,FlowID,Throughput_Mbps" << std::endl;
    rttStream << "Time,NodeID,RTT_ms" << std::endl;
    cwndStream << "Time,NodeID,Cwnd_Bytes" << std::endl;

    // --- TOPOLOGY: Same as tcp-evolution ---
    NodeContainer senders, receivers, gateways;
    senders.Create (6);
    receivers.Create (6);
    gateways.Create (2);

    PointToPointHelper p2pAccess;
    p2pAccess.SetDeviceAttribute ("DataRate", StringValue ("100Mbps"));
    p2pAccess.SetChannelAttribute ("Delay", StringValue ("1ms"));

    PointToPointHelper p2pBottleneck;
    p2pBottleneck.SetDeviceAttribute ("DataRate", StringValue ("10Mbps"));
    p2pBottleneck.SetChannelAttribute ("Delay", StringValue ("20ms"));
    p2pBottleneck.SetQueue ("ns3::DropTailQueue", "MaxSize", StringValue ("100p"));

    NetDeviceContainer bottleneckDevs = p2pBottleneck.Install (gateways.Get(0), gateways.Get(1));
    bottleneckDevs.Get(0)->TraceConnectWithoutContext("PhyTxDrop", MakeCallback (&RxDrop));

    std::vector<NetDeviceContainer> senderLinks;
    for (int i=0; i<6; i++) {
        senderLinks.push_back(p2pAccess.Install(senders.Get(i), gateways.Get(0)));
    }

    std::vector<NetDeviceContainer> receiverLinks;
    for (int i=0; i<6; i++) {
        receiverLinks.push_back(p2pAccess.Install(receivers.Get(i), gateways.Get(1)));
    }

    InternetStackHelper stack;
    stack.Install (senders);
    stack.Install (gateways);
    stack.Install (receivers);

    // --- IP ADDRESSING ---
    Ipv4AddressHelper ipv4;
    ipv4.SetBase ("10.1.1.0", "255.255.255.0");
    ipv4.Assign (bottleneckDevs);

    for (int i=0; i<6; i++) {
        std::stringstream ss;
        ss << "10.1." << (2+i) << ".0";
        ipv4.SetBase (ss.str().c_str(), "255.255.255.0");
        ipv4.Assign (senderLinks[i]);
    }

    std::vector<Ipv4InterfaceContainer> rxIfaces;
    for (int i=0; i<6; i++) {
        std::stringstream ss;
        ss << "10.1." << (10+i) << ".0";
        ipv4.SetBase (ss.str().c_str(), "255.255.255.0");
        rxIfaces.push_back(ipv4.Assign (receiverLinks[i]));
    }

    Ipv4GlobalRoutingHelper::PopulateRoutingTables ();

    // --- TRAFFIC GENERATION ---
    double startTimes[6] = {1.0, 2.0, 5.0, 10.0, 15.0, 20.0};
    
    for (int i=0; i<6; i++) {
        uint16_t port = 8000 + i;
        Address sinkAddr (InetSocketAddress (rxIfaces[i].GetAddress(0), port));
        
        PacketSinkHelper sink ("ns3::TcpSocketFactory", InetSocketAddress (Ipv4Address::GetAny (), port));
        ApplicationContainer sinkApp = sink.Install (receivers.Get(i));
        sinkApp.Start (Seconds (0.0));
        sinkApp.Stop (Seconds (60.0));
        sinkApp.Get(0)->TraceConnectWithoutContext("Rx", MakeBoundCallback(&PacketRx, i));

        BulkSendHelper source ("ns3::TcpSocketFactory", sinkAddr);
        source.SetAttribute ("MaxBytes", UintegerValue (0));
        ApplicationContainer sourceApp = source.Install (senders.Get(i));
        sourceApp.Start (Seconds (startTimes[i]));
        sourceApp.Stop (Seconds (60.0));
        
        Simulator::Schedule (Seconds (startTimes[i] + 0.001),
                            &ConnectSocketTraces,
                            senders.Get(i)->GetId(),
                            (uint32_t)i);
    }

    // --- OPENGYM SETUP ---
    openGymInterface = CreateObject<OpenGymInterface> (openGymPort);
    openGymInterface->SetGetActionSpaceCb (MakeCallback (&GetActionSpace));
    openGymInterface->SetGetObservationSpaceCb (MakeCallback (&GetObservationSpace));
    openGymInterface->SetGetObservationCb (MakeCallback (&GetObservation));
    openGymInterface->SetGetRewardCb (MakeCallback (&GetReward));
    openGymInterface->SetGetGameOverCb (MakeCallback (&GetGameOver));
    openGymInterface->SetGetExtraInfoCb (MakeCallback (&GetExtraInfo));
    openGymInterface->SetExecuteActionsCb (MakeCallback (&ExecuteAction));

    std::cout << "\n======================================================" << std::endl;
    std::cout << "   DRNN SIMULATION STARTING (port " << openGymPort << ")" << std::endl;
    std::cout << "   START YOUR PYTHON AGENT NOW!" << std::endl;
    std::cout << "======================================================\n" << std::endl;

    Simulator::Schedule (Seconds (0.1), &CalculateThroughput);
    // Agent step every 0.1s (matches one RTT ~42ms, gives ~10 decisions/sec)
    Simulator::Schedule (Seconds (1.0), &ScheduleNextStateRead, 0.1, openGymInterface);

    Simulator::Stop (Seconds (60.0));
    Simulator::Run ();
    
    openGymInterface->NotifySimulationEnd();
    
    throughputStream.close();
    rttStream.close();
    cwndStream.close();
    
    Simulator::Destroy ();
    return 0;
}
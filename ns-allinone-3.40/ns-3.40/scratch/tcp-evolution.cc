/*
 * TCP Evolution: Multi-Flow Dumbbell Topology
 * Features:
 * - 6 Senders vs 6 Receivers
 * - Shared Bottleneck (Competition)
 * - Data Collection: Throughput (Mbps), RTT (ms), Packet Loss (Count)
 */

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/applications-module.h"
#include "ns3/flow-monitor-module.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("TcpEvolution");

// --- DATA COLLECTION VARIABLES ---
std::ofstream throughputStream;
std::ofstream rttStream;
std::ofstream cwndStream;

// Map to store bytes received per flow for throughput calc
std::map<uint32_t, uint32_t> bytesReceived; 
std::map<uint32_t, uint32_t> lastBytesReceived;

// --- TRACE CALLBACKS ---

// 1. CWND Tracer
void CwndTracer (uint32_t nodeId, uint32_t socketId, uint32_t oldVal, uint32_t newVal) {
    // Format: Time, NodeID, CwndBytes
    cwndStream << Simulator::Now().GetSeconds() << "," << nodeId << "," << newVal << std::endl;
}

// 2. RTT Tracer
void RttTracer (uint32_t nodeId, uint32_t socketId, Time oldVal, Time newVal) {
    // Format: Time, NodeID, RTT(ms)
    rttStream << Simulator::Now().GetSeconds() << "," << nodeId << "," << newVal.GetMilliSeconds() << std::endl;
}

// 3. Packet Loss Tracer (Phy Drop)
void RxDrop (Ptr<const Packet> p) {
    NS_LOG_UNCOND("Packet Loss at " << Simulator::Now().GetSeconds());
}

void InFlightTracer (uint32_t nodeId, uint32_t socketId, uint32_t oldVal, uint32_t newVal) {
    // You can uncomment this if you want to track flight size
    // std::cout << "Node " << nodeId << " InFlight: " << newVal << std::endl;
}

// This function runs AFTER the app starts so the socket exists
void ConnectSocketTraces (uint32_t nodeId, uint32_t flowId) {
    std::stringstream pathCwnd, pathRtt;
    pathCwnd << "/NodeList/" << nodeId << "/$ns3::TcpL4Protocol/SocketList/*/CongestionWindow";
    pathRtt  << "/NodeList/" << nodeId << "/$ns3::TcpL4Protocol/SocketList/*/RTT";

    Config::ConnectWithoutContext (pathCwnd.str(), MakeBoundCallback (&CwndTracer, flowId, (uint32_t)0));
    Config::ConnectWithoutContext (pathRtt.str(),  MakeBoundCallback (&RttTracer,  flowId, (uint32_t)0));

    std::cout << "DEBUG: Traces attached for flow " << flowId << " (node " << nodeId << ") at " << Simulator::Now().GetSeconds() << "s" << std::endl;
}

// 4. Throughput Calculator (Runs every 0.1s)
void CalculateThroughput () {
    double now = Simulator::Now().GetSeconds();
    
    // We iterate over all 6 flows
    for (int i=0; i<6; i++) {
        uint32_t currentBytes = bytesReceived[i];
        uint32_t diff = currentBytes - lastBytesReceived[i];
        
        // Mbps = (Bytes * 8) / (0.1s * 10^6)
        double throughput = (diff * 8.0) / (0.1 * 1000000.0); 
        
        lastBytesReceived[i] = currentBytes;
        
        // Format: Time, FlowID, Throughput(Mbps)
        throughputStream << now << "," << i << "," << throughput << std::endl;
    }
    
    Simulator::Schedule (Seconds (0.1), &CalculateThroughput);
}

// Packet Received Callback (Increments counter)
void PacketRx (uint32_t flowId, Ptr<const Packet> p, const Address &addr) {
    bytesReceived[flowId] += p->GetSize();
}


int main (int argc, char *argv[]) {
    
    std::string transport_prot = "TcpCubic"; // Default
    std::string runName = "cubic";           // For file naming
    
    CommandLine cmd;
    cmd.AddValue ("transport", "Transport protocol: TcpNewReno, TcpCubic", transport_prot);
    cmd.AddValue ("name", "File name prefix (e.g., cubic, reno)", runName);
    cmd.Parse (argc, argv);

    // Set Protocol
    TypeId tcpTid = TypeId::LookupByName ("ns3::" + transport_prot);
    Config::SetDefault ("ns3::TcpL4Protocol::SocketType", TypeIdValue (tcpTid));

    // Open Data Files
    throughputStream.open (runName + "_throughput.csv");
    rttStream.open (runName + "_rtt.csv");
    cwndStream.open (runName + "_cwnd.csv");
    
    // Write Headers
    throughputStream << "Time,FlowID,Throughput_Mbps" << std::endl;
    rttStream << "Time,NodeID,RTT_ms" << std::endl;
    cwndStream << "Time,NodeID,Cwnd_Bytes" << std::endl;

    // --- TOPOLOGY: 6 Senders --(GW1)-- Bottleneck --(GW2)-- 6 Receivers ---
    
    NodeContainer senders, receivers;
    NodeContainer gateways;
    
    senders.Create (6);
    receivers.Create (6);
    gateways.Create (2); // GW1 (Left), GW2 (Right)

    // 1. Access Links (Fast)
    PointToPointHelper p2pAccess;
    p2pAccess.SetDeviceAttribute ("DataRate", StringValue ("100Mbps"));
    p2pAccess.SetChannelAttribute ("Delay", StringValue ("1ms"));

    // 2. Bottleneck Link (Slow & Shared)
    PointToPointHelper p2pBottleneck;
    p2pBottleneck.SetDeviceAttribute ("DataRate", StringValue ("10Mbps")); // 6 users fighting for 10Mbps!
    p2pBottleneck.SetChannelAttribute ("Delay", StringValue ("20ms"));
    // Add a Queue to cause buffering
    p2pBottleneck.SetQueue ("ns3::DropTailQueue", "MaxSize", StringValue ("100p"));

    // Install Infrastructure
    NetDeviceContainer bottleneckDevs = p2pBottleneck.Install (gateways.Get(0), gateways.Get(1));
    
    // Capture Packet Drops at the Bottleneck (Queue Overflow)
    bottleneckDevs.Get(0)->TraceConnectWithoutContext("PhyTxDrop", MakeCallback (&RxDrop));

    // Connect Senders to GW1
    std::vector<NetDeviceContainer> senderLinks;
    for (int i=0; i<6; i++) {
        senderLinks.push_back(p2pAccess.Install(senders.Get(i), gateways.Get(0)));
    }

    // Connect Receivers to GW2
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
    
    // Bottleneck Network
    ipv4.SetBase ("10.1.1.0", "255.255.255.0");
    ipv4.Assign (bottleneckDevs);

    // Sender Networks (10.1.2.0, 10.1.3.0, etc.)
    for (int i=0; i<6; i++) {
        std::stringstream ss;
        ss << "10.1." << (2+i) << ".0"; 
        ipv4.SetBase (ss.str().c_str(), "255.255.255.0");
        ipv4.Assign (senderLinks[i]);
    }

    // Receiver Networks (10.1.10.0, etc.)
    std::vector<Ipv4InterfaceContainer> rxIfaces;
    for (int i=0; i<6; i++) {
        std::stringstream ss;
        ss << "10.1." << (10+i) << ".0"; 
        ipv4.SetBase (ss.str().c_str(), "255.255.255.0");
        rxIfaces.push_back(ipv4.Assign (receiverLinks[i]));
    }

    Ipv4GlobalRoutingHelper::PopulateRoutingTables ();

    // --- TRAFFIC GENERATION (6 Flows) ---
    // Staggered Start Times to simulate real users joining
    double startTimes[6] = {1.0, 2.0, 5.0, 10.0, 15.0, 20.0};
    
    for (int i=0; i<6; i++) {
        uint16_t port = 8000 + i;
        Address sinkAddr (InetSocketAddress (rxIfaces[i].GetAddress(0), port));
        
        // Sink (Receiver)
        PacketSinkHelper sink ("ns3::TcpSocketFactory", InetSocketAddress (Ipv4Address::GetAny (), port));
        ApplicationContainer sinkApp = sink.Install (receivers.Get(i));
        sinkApp.Start (Seconds (0.0));
        sinkApp.Stop (Seconds (60.0));
        
        // Hook into Sink to count bytes for throughput
        sinkApp.Get(0)->TraceConnectWithoutContext("Rx", MakeBoundCallback(&PacketRx, i));

        // Source (Sender)
        BulkSendHelper source ("ns3::TcpSocketFactory", sinkAddr);
        source.SetAttribute ("MaxBytes", UintegerValue (0));
        ApplicationContainer sourceApp = source.Install (senders.Get(i));
        sourceApp.Start (Seconds (startTimes[i])); // Staggered Start
        sourceApp.Stop (Seconds (60.0));
        
        // Schedule trace connection AFTER the app starts (socket must exist first)
        Simulator::Schedule (Seconds (startTimes[i] + 0.001),
                            &ConnectSocketTraces,
                            senders.Get(i)->GetId(),
                            (uint32_t)i);
    }

    // Schedule Throughput Calculation
    Simulator::Schedule (Seconds (0.1), &CalculateThroughput);

    Simulator::Stop (Seconds (60.0));
    Simulator::Run ();
    
    throughputStream.close();
    rttStream.close();
    cwndStream.close();
    
    Simulator::Destroy ();
    return 0;
}
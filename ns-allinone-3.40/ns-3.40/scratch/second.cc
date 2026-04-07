#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/csma-module.h"
#include "ns3/internet-module.h"
#include "ns3/ipv4-global-routing-helper.h"
#include "ns3/network-module.h"
#include "ns3/point-to-point-module.h"



using namespace ns3;
NS_LOG_COMPONENT_DEFINE("SecondScriptExample");

int main(int argc, char* argv[]){

    bool verbose = true;
    int nCsma = 3;
    uint32_t PORT = 9;
    CommandLine cmd(__FILE__);

    cmd.addValue("nCsma", "Add csma to the topology", nCsma);
    cmd.addValue("verbose", "Tell echo to log if true", verbose);

    cmd.Parse(argc, argv);

    if(verbose){
        LogComponentEnable("UdpEchoClientApplication", LOG_LEVEL_INFO);
        LogComponentEnable("UdpEchoServerApplication", LOG_LEVEL_INFO);

    }


    nCsma = nCsma > 0? nCsma: 1;

    NodeContainer p2pNodes;
    p2pNodes.Create(2);

    NodeContainer csmaNodes;
    csmaNodes.Add(p2pNodes.Get(1));
    csmaNodes.Create(nCsma);



    PointToPointHelper p2p;
    p2p.SetChannelAttribute("Delay", StringValue("2ms"));
    p2p.SetDeviceAttribute("DataRate", StringValue("5Mbps"));

    NetDeviceContainer p2pDevices;
    p2pDevices = p2p.Install(p2pNodes)

    CsmaHelper csma;
    csma.SetChannelAttribute("Delay", TimeValue(NanoSeconds(6560)));
    csma.SetChannelAttribute("DataRate", StringValue("100Mbps"));

    NetDeviceContainer csmaDevices;
    csmaDevices = csma.Install(csmaNodes);

    InternetStackHelper stack;
    stack.Install(p2pNodes.Get(0));
    stack.Install(csmaNodes);

    Ipv4AddressHelper address;
    address.SetBase("10.1.1.0", "255.255.255.0");

    Ipv4InterfaceContainer p2pInterface;
    p2pInterface = address.Assign(p2pDevices);

    address.SetBase("10.1.2.0", "255.255.255.0");

    Ipv4InterfaceContainer csmaInterface;
    csmaInterface = address.Assign(csmaInterface);

    UdpEchoServerHelper echoServer(PORT);

    ApplicationContainer serverApps = echoServer.Install(csmaNodes.Get(nCsma));

    serverApps.Start(Seconds(1.0));
    serverApps.Stop(Seconds(10.0));

    UdpEchoClientHelper echoClient(csmaInterface.GetAddress(nCsma), PORT);

    echoClient.SetAttribute("MaxPackets", UintegerValue(1));
    echoClient.SetAttribute("Interval", TimeValue(Seconds(1.0)));
    echoClient.SetAttribute("PacketSize", UintegerValue(1024));


    ApplicationContainer clientApps = echoClient.Install(p2pNodes.Get(0));

    clientApps.Start(Seconds(2.0));
    clientApps.Stop(Seconds(10.0));

    Ipv4GlobalRoutingHelper::PopulateRoutingTables();
    p2p.EnablePcapAll("scratch/second/second");
    csma.EnablePcap("second", csmaDevices.Get(1), true);

    Simulator::Run();
    Simulator::Destroy();
    return 0;



}
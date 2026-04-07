#include "ns3/application-module.h"
#include "ns3/core-module.h"
#include "ns3/internet-module.h"
#include "ns3/network-module.h"
#include "ns3/point-to-point-module.h"


using namespace ns3;

NS_LOG_COMPONENT_DEFINE("FirstExampleSimulator");

int main(int argc, char * argv[]){
    CommandLine cmd(__FILE__);
    cmd.Parse(argc, argv);

    uint32_t nPackets = 1;

    Time::SetResolution(Time::NS);

    LogComponentEnable("UdpEchoClientApplication", LOG_LEVEL_INFO);
    LogComponentEnable("UdpEchoServerApplication", LOG_LEVEL_INFO);

    NodeContainer nodes;
    nodes.Create(2);

    PointToPointHelper pointToPoint;

    pointToPoint.SetDeviceAttribute("DataRate", StringValue("5Mbps"));
    pointToPoint.SetChannelAttribute("Delat", StringValue("2ms"));

    NetDeviceContainer devices;
    devices = pointToPoint.Install(nodes);

    InternetStackHelper stack;
    stack.Install(nodes);

    Ipv4AddressHelper address;
    address.SetBase("10.1.1.0", "255.255.255.0");


    Ipv4InterfaceContainer interfaces = address.Assign(devices);

    UdpEchoServerHleper echoServer(9);

    ApplicationContainer serverApp = echoServer.install(nodes.Get(1));

    serverApp.Start(Seconds(1.0));
    serverApp.Stop(Seconds(10.0));

    // server part done ere

    UdpEchoClientHelper echoClient(interfaces.GetAddress(1), 9);

    ApplicationContainer clientApp = echoClient.install(nodes.Get(0));

    clientApp.SetAttribute("MaxPackets", UintegerValue(1));
    clientApp.SetAttribute("Interval", TimeValue(Seconds(1.0)));
    clientApp.SetAttribute("PacketSize", UintegerValue(1024));

    clientApp.Start(Seconds(2.0));
    clientApp.Stop(Seconds(10.0));

    Simulator::Stop(Seconds(10.0)):

    Simulator::Run();

    Simulator::Destroy();




}
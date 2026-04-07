#include "ns3/aodv-module.h"
#include "ns3/aodv-packet.h"
#include "ns3/aodv-routing-protocol.h"
#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/energy-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/internet-module.h"
#include "ns3/ipv4-list-routing.h"
#include "ns3/mobility-module.h"
#include "ns3/network-module.h"
#include "ns3/udp-header.h"
#include "ns3/wifi-module.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using namespace ns3;
using namespace ns3::energy;
using namespace std;

NS_LOG_COMPONENT_DEFINE("AodvSimulator");

struct SimulationOptions
{
    uint32_t nNodes = 30;
    double simTime = 100.0;
    double speed = -1.0;
    double minSpeed = 0.0;
    double maxSpeed = 5.0;
    uint32_t nSinks = 10;
    uint32_t seed = 1;
    string output = "aodv";
    string mode = "aodv";
    bool trace = true;
    uint32_t pcapNodes = 5;
    string asciiNodes = "all";
    int32_t flowSrcNode = -1;
    int32_t flowDstNode = -1;
    bool routeDiscoveryTrace = true;
    bool routingTableTrace = true;
    double routingTableInterval = 1.0;
    string outputDir = "outputs";
    double flowMonitorCleanupTime = 10.0;
    uint32_t ccBaseThreshold = 4;
    double w1 = 0.5;
    double w2 = 0.3;
    double w3 = 0.2;
    // Traffic parameters
    uint32_t pps = 0;           // packets per second (0 = use default 4 pps)
    uint32_t packetSize = 512;  // bytes per packet
    // Network type
    string networkType = "802.11"; // "802.11" or "802.15.4"
};

struct Metrics
{
    uint64_t totalTx = 0;
    uint64_t totalRx = 0;
    uint64_t totalLost = 0;
    uint64_t totalBytes = 0;
    double totalDelay = 0.0;
    double pdr = 0.0;
    double lossRate = 0.0;
    double avgDelayMs = 0.0;
    double throughputKbps = 0.0;
    double totalEnergyJ = 0.0;       // Total energy consumed (Joules) across all nodes
    double avgEnergyPerNodeJ = 0.0;  // Average energy per node (Joules)
};

struct RouteEntry
{
    string nextHop;
    uint32_t hops = 0;
};

struct SimTraceContext
{
    bool enabled = false;
    string traceBase;
    string asciiFile;
    vector<uint32_t> asciiNodeIds;
    string ipv4DropFile;
    string rreqFile;
    string rrepFile;
    string routingTableFile;
};

static uint64_t g_ipv4DropCount = 0;
static uint64_t g_rreqCount = 0;
static uint64_t g_rrepCount = 0;
static map<uint32_t, map<string, RouteEntry>> g_lastRouteTable;
static Ipv4Address g_flowTraceSrc;
static Ipv4Address g_flowTraceDst;
static bool g_flowTraceEnabled = false;

string
Trim(const string& text)
{
    size_t left = text.find_first_not_of(" \t\n\r");
    if (left == string::npos)
    {
        return "";
    }
    size_t right = text.find_last_not_of(" \t\n\r");
    return text.substr(left, right - left + 1);
}

uint32_t
ParseNodeIdFromContext(const string& context)
{
    smatch match;
    regex re("/NodeList/([0-9]+)/");
    if (!regex_search(context, match, re) || match.size() < 2)
    {
        return numeric_limits<uint32_t>::max();
    }
    return static_cast<uint32_t>(stoul(match[1]));
}

vector<uint32_t>
ParseAsciiNodeList(const string& value, uint32_t nNodes)
{
    string raw = Trim(value);
    if (raw.empty() || raw == "all" || raw == "ALL")
    {
        vector<uint32_t> ids;
        ids.reserve(nNodes);
        for (uint32_t i = 0; i < nNodes; i++)
        {
            ids.push_back(i);
        }
        return ids;
    }

    set<uint32_t> uniqueIds;
    string token;
    stringstream ss(raw);
    while (getline(ss, token, ','))
    {
        token = Trim(token);
        if (token.empty())
        {
            cerr << "Invalid --asciiNodes: empty token in list\n";
            exit(1);
        }
        for (char ch : token)
        {
            if (!isdigit(static_cast<unsigned char>(ch)))
            {
                cerr << "Invalid --asciiNodes token: " << token << "\n";
                exit(1);
            }
        }
        uint32_t id = static_cast<uint32_t>(stoul(token));
        if (id >= nNodes)
        {
            cerr << "Invalid --asciiNodes node id " << id << " for node count " << nNodes << "\n";
            exit(1);
        }
        uniqueIds.insert(id);
    }

    if (uniqueIds.empty())
    {
        cerr << "Invalid --asciiNodes: no valid node ids\n";
        exit(1);
    }

    return vector<uint32_t>(uniqueIds.begin(), uniqueIds.end());
}

static void
Ipv4DropLogger(Ptr<OutputStreamWrapper> stream,
               string context,
               const Ipv4Header& header,
               Ptr<const Packet> packet,
               Ipv4L3Protocol::DropReason reason,
               Ptr<Ipv4>,
               uint32_t interface)
{
    g_ipv4DropCount++;
    *stream->GetStream() << fixed << setprecision(6) << Simulator::Now().GetSeconds() << ","
                         << context << "," << header.GetSource() << "," << header.GetDestination()
                         << "," << packet->GetSize() << "," << static_cast<uint32_t>(reason) << ","
                         << interface << "\n";
}

static void
MobilityCourseChangeLogger(Ptr<OutputStreamWrapper> stream,
                           string context,
                           Ptr<const MobilityModel> model)
{
    Vector pos = model->GetPosition();
    Vector vel = model->GetVelocity();
    *stream->GetStream() << fixed << setprecision(6) << Simulator::Now().GetSeconds() << ","
                         << ParseNodeIdFromContext(context) << "," << pos.x << "," << pos.y << ","
                         << pos.z << "," << vel.x << "," << vel.y << "," << vel.z << "\n";
}

static bool
DecodeAodvControl(Ptr<const Packet> packet,
                  Ipv4Header& ipv4Header,
                  ns3::aodv::TypeHeader& type,
                  ns3::aodv::RreqHeader& rreq,
                  ns3::aodv::RrepHeader& rrep)
{
    Ptr<Packet> copy = packet->Copy();
    if (copy->PeekHeader(ipv4Header) == 0)
    {
        return false;
    }
    copy->RemoveHeader(ipv4Header);
    if (ipv4Header.GetProtocol() != 17)
    {
        return false;
    }

    UdpHeader udp;
    if (!copy->PeekHeader(udp))
    {
        return false;
    }
    uint32_t aodvPort = ns3::aodv::RoutingProtocol::AODV_PORT;
    if (udp.GetDestinationPort() != aodvPort && udp.GetSourcePort() != aodvPort)
    {
        return false;
    }
    copy->RemoveHeader(udp);

    if (!copy->PeekHeader(type))
    {
        return false;
    }
    copy->RemoveHeader(type);

    if (type.Get() == ns3::aodv::AODVTYPE_RREQ)
    {
        return copy->PeekHeader(rreq);
    }
    if (type.Get() == ns3::aodv::AODVTYPE_RREP)
    {
        return copy->PeekHeader(rrep);
    }
    return false;
}

static bool
FlowMatches(const Ipv4Address& src, const Ipv4Address& dst)
{
    if (!g_flowTraceEnabled)
    {
        return true;
    }
    return src == g_flowTraceSrc && dst == g_flowTraceDst;
}

static void
RreqLogger(Ptr<OutputStreamWrapper> stream,
           string context,
           Ptr<const Packet> packet,
           Ptr<Ipv4>,
           uint32_t interface)
{
    Ipv4Header ipv4Header;
    ns3::aodv::TypeHeader type;
    ns3::aodv::RreqHeader rreq;
    ns3::aodv::RrepHeader rrep;
    if (!DecodeAodvControl(packet, ipv4Header, type, rreq, rrep))
    {
        return;
    }
    if (type.Get() != ns3::aodv::AODVTYPE_RREQ)
    {
        return;
    }
    if (!FlowMatches(rreq.GetOrigin(), rreq.GetDst()))
    {
        return;
    }

    uint32_t nodeId = ParseNodeIdFromContext(context);
    if (nodeId == numeric_limits<uint32_t>::max())
    {
        return;
    }

    g_rreqCount++;
    *stream->GetStream() << fixed << setprecision(6) << Simulator::Now().GetSeconds() << ","
                         << nodeId << "," << rreq.GetOrigin() << "," << rreq.GetDst() << ","
                         << static_cast<uint32_t>(rreq.GetHopCount()) << "," << interface << "\n";
}

static void
RrepLogger(Ptr<OutputStreamWrapper> stream,
           string context,
           Ptr<const Packet> packet,
           Ptr<Ipv4>,
           uint32_t interface)
{
    Ipv4Header ipv4Header;
    ns3::aodv::TypeHeader type;
    ns3::aodv::RreqHeader rreq;
    ns3::aodv::RrepHeader rrep;
    if (!DecodeAodvControl(packet, ipv4Header, type, rreq, rrep))
    {
        return;
    }
    if (type.Get() != ns3::aodv::AODVTYPE_RREP)
    {
        return;
    }
    if (rrep.GetOrigin() == rrep.GetDst())
    {
        return;
    }
    if (!FlowMatches(rrep.GetOrigin(), rrep.GetDst()))
    {
        return;
    }

    uint32_t nodeId = ParseNodeIdFromContext(context);
    if (nodeId == numeric_limits<uint32_t>::max())
    {
        return;
    }

    g_rrepCount++;
    *stream->GetStream() << fixed << setprecision(6) << Simulator::Now().GetSeconds() << ","
                         << nodeId << "," << rrep.GetOrigin() << "," << rrep.GetDst() << ","
                         << static_cast<uint32_t>(rrep.GetHopCount()) << "," << interface << "\n";
}

Ptr<ns3::aodv::RoutingProtocol>
GetAodvRoutingProtocol(Ptr<Node> node)
{
    Ptr<Ipv4> ipv4 = node->GetObject<Ipv4>();
    if (!ipv4)
    {
        return nullptr;
    }

    Ptr<Ipv4RoutingProtocol> routing = ipv4->GetRoutingProtocol();
    if (!routing)
    {
        return nullptr;
    }

    Ptr<ns3::aodv::RoutingProtocol> direct = DynamicCast<ns3::aodv::RoutingProtocol>(routing);
    if (direct)
    {
        return direct;
    }

    Ptr<Ipv4ListRouting> list = DynamicCast<Ipv4ListRouting>(routing);
    if (!list)
    {
        return nullptr;
    }

    for (uint32_t i = 0; i < list->GetNRoutingProtocols(); i++)
    {
        int16_t priority = 0;
        Ptr<Ipv4RoutingProtocol> item = list->GetRoutingProtocol(i, priority);
        Ptr<ns3::aodv::RoutingProtocol> aodv = DynamicCast<ns3::aodv::RoutingProtocol>(item);
        if (aodv)
        {
            return aodv;
        }
    }

    return nullptr;
}

map<string, RouteEntry>
ParseRoutingTableDump(const string& dump)
{
    map<string, RouteEntry> table;
    regex ipRegex("([0-9]+\\.[0-9]+\\.[0-9]+\\.[0-9]+)");
    regex hopRegex("([0-9]+)\\s*$");

    istringstream in(dump);
    string line;
    while (getline(in, line))
    {
        vector<string> ips;
        for (sregex_iterator it(line.begin(), line.end(), ipRegex), end; it != end; ++it)
        {
            ips.push_back((*it)[1]);
        }
        if (ips.size() < 2)
        {
            continue;
        }

        smatch hopMatch;
        if (!regex_search(line, hopMatch, hopRegex) || hopMatch.size() < 2)
        {
            continue;
        }

        RouteEntry entry;
        entry.nextHop = ips[1];
        entry.hops = static_cast<uint32_t>(stoul(hopMatch[1]));
        table[ips[0]] = entry;
    }

    return table;
}

void
PollRoutingTables(vector<Ptr<Node>> nodes,
                  Ptr<OutputStreamWrapper> stream,
                  Time interval,
                  Time stopTime)
{
    for (const auto& node : nodes)
    {
        if (!node)
        {
            continue;
        }

        uint32_t nodeId = node->GetId();
        Ptr<ns3::aodv::RoutingProtocol> aodv = GetAodvRoutingProtocol(node);
        if (!aodv)
        {
            continue;
        }

        ostringstream dump;
        Ptr<OutputStreamWrapper> temp = Create<OutputStreamWrapper>(&dump);
        aodv->PrintRoutingTable(temp, Time::S);

        map<string, RouteEntry> current = ParseRoutingTableDump(dump.str());
        map<string, RouteEntry>& previous = g_lastRouteTable[nodeId];

        for (const auto& [dst, cur] : current)
        {
            auto it = previous.find(dst);
            if (it == previous.end())
            {
                *stream->GetStream()
                    << fixed << setprecision(6) << Simulator::Now().GetSeconds() << "," << nodeId
                    << ",added," << dst << "," << cur.nextHop << "," << cur.hops << "\n";
            }
            else if (it->second.nextHop != cur.nextHop || it->second.hops != cur.hops)
            {
                *stream->GetStream()
                    << fixed << setprecision(6) << Simulator::Now().GetSeconds() << "," << nodeId
                    << ",updated," << dst << "," << cur.nextHop << "," << cur.hops << "\n";
            }
        }

        for (const auto& [dst, prev] : previous)
        {
            if (current.find(dst) == current.end())
            {
                *stream->GetStream()
                    << fixed << setprecision(6) << Simulator::Now().GetSeconds() << "," << nodeId
                    << ",removed," << dst << "," << prev.nextHop << "," << prev.hops << "\n";
            }
        }

        previous = current;
    }

    if (Simulator::Now() + interval <= stopTime)
    {
        Simulator::Schedule(interval, &PollRoutingTables, nodes, stream, interval, stopTime);
    }
}

SimulationOptions
ParseArgs(int argc, char* argv[])
{
    SimulationOptions opt;

    CommandLine cmd(__FILE__);
    cmd.AddValue("nodes", "Number of nodes (e.g. 10, 20, 30)", opt.nNodes);
    cmd.AddValue("time", "Simulation time in seconds", opt.simTime);
    cmd.AddValue("speed", "[Deprecated] Sets maxSpeed", opt.speed);
    cmd.AddValue("minSpeed", "Min node speed in m/s", opt.minSpeed);
    cmd.AddValue("maxSpeed", "Max node speed in m/s", opt.maxSpeed);
    cmd.AddValue("sinks", "Number of traffic flows (src->dst pairs)", opt.nSinks);
    cmd.AddValue("seed", "Random seed for reproducibility", opt.seed);
    cmd.AddValue("output", "Output CSV filename prefix", opt.output);
    cmd.AddValue("mode", "Protocol mode: aodv | cc-aodv | ecc-aodv", opt.mode);
    cmd.AddValue("trace", "Enable tracing", opt.trace);
    cmd.AddValue("pcapNodes", "Number of nodes to capture PCAP on", opt.pcapNodes);
    cmd.AddValue("asciiNodes",
                 "ASCII PHY trace node list: all or comma-separated ids",
                 opt.asciiNodes);
    cmd.AddValue("routeDiscoveryTrace", "Enable route discovery trace", opt.routeDiscoveryTrace);
    cmd.AddValue("routingTableTrace", "Enable routing table change trace", opt.routingTableTrace);
    cmd.AddValue("routingTableInterval",
                 "Routing table polling interval in seconds",
                 opt.routingTableInterval);
    cmd.AddValue("flowMonitorCleanupTime",
                 "Extra seconds to run after traffic stops (for FlowMonitor loss accounting)",
                 opt.flowMonitorCleanupTime);
    cmd.AddValue("ccBaseThreshold",
                 "CC-AODV congestion threshold used when mode is cc-aodv/ecc-aodv",
                 opt.ccBaseThreshold);
    cmd.AddValue("w1", "ECC-AODV QACD weight for queue occupancy", opt.w1);
    cmd.AddValue("w2", "ECC-AODV QACD weight for congestion counter", opt.w2);
    cmd.AddValue("w3", "ECC-AODV QACD weight for drop rate", opt.w3);
    cmd.AddValue("pps", "Packets per second per flow (0=use default 4 pps)", opt.pps);
    cmd.AddValue("packetSize", "UDP payload size in bytes", opt.packetSize);
    cmd.AddValue("networkType", "Network type: 802.11 or 802.15.4", opt.networkType);
    cmd.AddValue("outputDir", "Directory to write output CSVs and traces", opt.outputDir);
    cmd.AddValue("flowSrcNode",
                 "Source node for route discovery filter (-1 disables filter)",
                 opt.flowSrcNode);
    cmd.AddValue("flowDstNode",
                 "Destination node for route discovery filter (-1 disables filter)",
                 opt.flowDstNode);
    cmd.Parse(argc, argv);

    if (opt.speed >= 0.0)
    {
        opt.maxSpeed = opt.speed;
    }

    if (opt.nNodes < 2)
    {
        cerr << "Error: nodes must be >= 2\n";
        exit(1);
    }
    if (opt.routingTableInterval <= 0.0)
    {
        cerr << "Error: routingTableInterval must be > 0\n";
        exit(1);
    }
    if (opt.flowMonitorCleanupTime < 0.0)
    {
        cerr << "Error: flowMonitorCleanupTime must be >= 0\n";
        exit(1);
    }
    if (opt.minSpeed < 0.0 || opt.maxSpeed < 0.0 || opt.minSpeed > opt.maxSpeed)
    {
        cerr << "Error: require 0 <= minSpeed <= maxSpeed\n";
        exit(1);
    }
    if ((opt.flowSrcNode == -1) != (opt.flowDstNode == -1))
    {
        cerr << "Error: flowSrcNode and flowDstNode must both be set or both be -1\n";
        exit(1);
    }
    if (opt.flowSrcNode < -1 || opt.flowDstNode < -1)
    {
        cerr << "Error: flowSrcNode/flowDstNode must be >= -1\n";
        exit(1);
    }
    if (opt.flowSrcNode >= static_cast<int32_t>(opt.nNodes) ||
        opt.flowDstNode >= static_cast<int32_t>(opt.nNodes))
    {
        cerr << "Error: flowSrcNode/flowDstNode out of range for nodes\n";
        exit(1);
    }

    opt.nSinks = min(opt.nSinks, opt.nNodes);
    opt.pcapNodes = min(opt.pcapNodes, opt.nNodes);

    return opt;
}

string
ResolveProtocolLabel(const string& mode)
{
    if (mode == "aodv")
    {
        return "Standard AODV";
    }
    if (mode == "cc-aodv")
    {
        return "CC-AODV";
    }
    if (mode == "ecc-aodv")
    {
        return "ECC-AODV";
    }

    cerr << "Unknown mode: " << mode << ". Use aodv | cc-aodv | ecc-aodv\n";
    exit(1);
}

void
EnsureOutputDir(const SimulationOptions& opt)
{
    filesystem::create_directories(opt.outputDir);
}

void
PrintConfig(const SimulationOptions& opt,
            const string& protocolLabel,
            const vector<uint32_t>& asciiNodes)
{
    cout << "\n========================================\n";
    cout << "  AODV SIMULATOR\n";
    cout << "========================================\n";
    cout << "Protocol:  " << protocolLabel << "\n";
    cout << "Mode:      " << opt.mode << "\n";
    cout << "Nodes:     " << opt.nNodes << "\n";
    cout << "SimTime:   " << opt.simTime << " s\n";
    cout << "Speed:     " << opt.minSpeed << ".." << opt.maxSpeed << " m/s\n";
    cout << "Flows:     " << opt.nSinks << "\n";
    cout << "Seed:      " << opt.seed << "\n";
    cout << "Trace:     " << (opt.trace ? "true" : "false") << "\n";
    cout << "ASCII PHY nodes: ";
    for (size_t i = 0; i < asciiNodes.size(); i++)
    {
        cout << asciiNodes[i];
        if (i + 1 < asciiNodes.size())
        {
            cout << ",";
        }
    }
    cout << "\n";
    if (opt.flowSrcNode >= 0)
    {
        cout << "Flow trace: src=" << opt.flowSrcNode << " dst=" << opt.flowDstNode << "\n";
    }
    else
    {
        cout << "Flow trace: all src-dst pairs\n";
    }
    cout << "Output:    " << opt.outputDir << "/" << opt.output << "-results.csv\n";
    cout << "========================================\n\n";
}

void
ConfigureRoutingForMode(AodvHelper& aodv, const SimulationOptions& opt)
{
    const string& mode = opt.mode;
    aodv.Set("EnableCcAodv", BooleanValue(false));
    aodv.Set("EnableEccAodv", BooleanValue(false));
    aodv.Set("BaseThreshold", UintegerValue(opt.ccBaseThreshold));

    if (mode == "aodv")
    {
    }
    else if (mode == "cc-aodv")
    {
        aodv.Set("EnableCcAodv", BooleanValue(true));
        cout << "CC-AODV enabled with BaseThreshold=" << opt.ccBaseThreshold << "\n";
    }
    else if (mode == "ecc-aodv")
    {
        aodv.Set("EnableCcAodv", BooleanValue(true));
        aodv.Set("EnableEccAodv", BooleanValue(true));
        aodv.Set("QACDWeightQueue", DoubleValue(opt.w1));
        aodv.Set("QACDWeightCounter", DoubleValue(opt.w2));
        aodv.Set("QACDWeightDropRate", DoubleValue(opt.w3));
        cout << "ECC-AODV enabled with BaseThreshold=" << opt.ccBaseThreshold
             << " and QACD weights (" << opt.w1 << "," << opt.w2 << "," << opt.w3 << ")\n";
    }
}

NetDeviceContainer
SetupWifi(NodeContainer& nodes, YansWifiPhyHelper& wifiPhy)
{
    WifiHelper wifi;
    wifi.SetStandard(WIFI_STANDARD_80211b);
    wifi.SetRemoteStationManager("ns3::ConstantRateWifiManager",
                                 "DataMode",
                                 StringValue("DsssRate11Mbps"),
                                 "ControlMode",
                                 StringValue("DsssRate1Mbps"));

    WifiMacHelper wifiMac;
    wifiMac.SetType("ns3::AdhocWifiMac");

    YansWifiChannelHelper wifiChannel;
    wifiChannel.SetPropagationDelay("ns3::ConstantSpeedPropagationDelayModel");
    wifiChannel.AddPropagationLoss("ns3::FriisPropagationLossModel");
    wifiPhy.SetChannel(wifiChannel.Create());

    return wifi.Install(wifiPhy, wifiMac, nodes);
}

// 802.15.4-approximated network
// Lower PHY data rate (1 Mbps) and limited range (200 m) via RangePropagationLossModel
// 802.15.4 outdoor LOS range is typically 75–200 m at max Tx power (0–5 dBm)
// Application PPS controls throughput to approximate 802.15.4 250 Kbps data rate
NetDeviceContainer
SetupLrWpanApprox(NodeContainer& nodes, YansWifiPhyHelper& wifiPhy)
{
    WifiHelper wifi;
    wifi.SetStandard(WIFI_STANDARD_80211b);
    wifi.SetRemoteStationManager("ns3::ConstantRateWifiManager",
                                 "DataMode",
                                 StringValue("DsssRate1Mbps"),
                                 "ControlMode",
                                 StringValue("DsssRate1Mbps"));

    WifiMacHelper wifiMac;
    wifiMac.SetType("ns3::AdhocWifiMac");

    YansWifiChannelHelper wifiChannel;
    wifiChannel.SetPropagationDelay("ns3::ConstantSpeedPropagationDelayModel");
    // Friis for path loss within range + range cap at 200 m (802.15.4 outdoor spec)
    wifiChannel.AddPropagationLoss("ns3::FriisPropagationLossModel");
    wifiChannel.AddPropagationLoss("ns3::RangePropagationLossModel",
                                   "MaxRange",
                                   DoubleValue(200.0));
    wifiPhy.SetChannel(wifiChannel.Create());

    return wifi.Install(wifiPhy, wifiMac, nodes);
}

EnergySourceContainer
SetupEnergy(NodeContainer& nodes, NetDeviceContainer& devices)
{
    BasicEnergySourceHelper energySourceHelper;
    energySourceHelper.Set("BasicEnergySourceInitialEnergyJ", DoubleValue(100.0));
    EnergySourceContainer sources = energySourceHelper.Install(nodes);

    WifiRadioEnergyModelHelper radioEnergyHelper;
    radioEnergyHelper.Install(devices, sources);

    return sources;
}

void
SetupMobility(const SimulationOptions& opt, NodeContainer& nodes)
{
    double gridWidth = ceil(sqrt((double)opt.nNodes));
    double spacing = 100.0;
    double areaSize = gridWidth * spacing;

    ostringstream speedStr, boundsX, boundsY;
    speedStr << "ns3::UniformRandomVariable[Min=" << opt.minSpeed << "|Max=" << opt.maxSpeed << "]";
    boundsX << "ns3::UniformRandomVariable[Min=0.0|Max=" << areaSize << "]";
    boundsY << "ns3::UniformRandomVariable[Min=0.0|Max=" << areaSize << "]";

    MobilityHelper mobility;
    mobility.SetPositionAllocator("ns3::GridPositionAllocator",
                                  "MinX",
                                  DoubleValue(0.0),
                                  "MinY",
                                  DoubleValue(0.0),
                                  "DeltaX",
                                  DoubleValue(spacing),
                                  "DeltaY",
                                  DoubleValue(spacing),
                                  "GridWidth",
                                  UintegerValue((uint32_t)gridWidth),
                                  "LayoutType",
                                  StringValue("RowFirst"));
    mobility.SetMobilityModel("ns3::RandomWaypointMobilityModel",
                              "Speed",
                              StringValue(speedStr.str()),
                              "Pause",
                              StringValue("ns3::ConstantRandomVariable[Constant=2.0]"),
                              "PositionAllocator",
                              StringValue("ns3::RandomRectanglePositionAllocator"));

    Config::SetDefault("ns3::RandomRectanglePositionAllocator::X", StringValue(boundsX.str()));
    Config::SetDefault("ns3::RandomRectanglePositionAllocator::Y", StringValue(boundsY.str()));
    mobility.Install(nodes);

    cout << "Topology: " << (uint32_t)gridWidth << "x" << (uint32_t)gridWidth << " grid, "
         << spacing << " m spacing, area " << areaSize << "x" << areaSize << " m\n";
}

Ipv4InterfaceContainer
SetupInternet(const SimulationOptions& opt, NodeContainer& nodes, NetDeviceContainer& devices)
{
    AodvHelper aodv;
    ConfigureRoutingForMode(aodv, opt);

    InternetStackHelper internet;
    internet.SetRoutingHelper(aodv);
    internet.Install(nodes);

    Ipv4AddressHelper address;
    address.SetBase("10.1.1.0", "255.255.255.0");
    return address.Assign(devices);
}

void
InstallTraffic(const SimulationOptions& opt,
               NodeContainer& nodes,
               const Ipv4InterfaceContainer& interfaces)
{
    uint16_t port = 9;
    uint32_t packetSize = opt.packetSize;
    // If pps specified use it; otherwise fall back to 4 pps default
    double packetRate = (opt.pps > 0) ? static_cast<double>(opt.pps) : 4.0;

    ApplicationContainer sourceApps;
    ApplicationContainer sinkApps;

    cout << "\nTraffic flows (" << opt.nSinks << "):\n";

    for (uint32_t i = 0; i < opt.nSinks; i++)
    {
        uint32_t sinkNode = opt.nNodes - 1 - i;
        Address sinkAddr(InetSocketAddress(interfaces.GetAddress(sinkNode), port));

        PacketSinkHelper sinkHelper("ns3::UdpSocketFactory",
                                    InetSocketAddress(Ipv4Address::GetAny(), port));
        sinkApps.Add(sinkHelper.Install(nodes.Get(sinkNode)));

        OnOffHelper src("ns3::UdpSocketFactory", sinkAddr);
        src.SetAttribute("OnTime", StringValue("ns3::ConstantRandomVariable[Constant=1]"));
        src.SetAttribute("OffTime", StringValue("ns3::ConstantRandomVariable[Constant=0]"));
        src.SetAttribute("DataRate", DataRateValue(DataRate(packetSize * 8 * packetRate)));
        src.SetAttribute("PacketSize", UintegerValue(packetSize));
        sourceApps.Add(src.Install(nodes.Get(i)));

        cout << "  Flow " << i << ": Node " << i << " -> Node " << sinkNode << "\n";
    }

    sinkApps.Start(Seconds(0.0));
    sinkApps.Stop(Seconds(opt.simTime + opt.flowMonitorCleanupTime));
    sourceApps.Start(Seconds(1.0));
    sourceApps.Stop(Seconds(opt.simTime));
}

SimTraceContext
SetupTracing(const SimulationOptions& opt,
             const vector<uint32_t>& asciiNodeIds,
             NodeContainer& nodes,
             YansWifiPhyHelper& wifiPhy,
             NetDeviceContainer& devices)
{
    SimTraceContext ctx;
    if (!opt.trace)
    {
        return ctx;
    }

    g_ipv4DropCount = 0;
    g_rreqCount = 0;
    g_rrepCount = 0;
    g_lastRouteTable.clear();

    ctx.enabled = true;
    ctx.traceBase =
        opt.outputDir + "/" + opt.output + "-" + opt.mode + "-" + to_string(opt.nNodes) + "nodes";
    ctx.asciiNodeIds = asciiNodeIds;

    AsciiTraceHelper ascii;

    if (asciiNodeIds.size() == opt.nNodes)
    {
        ctx.asciiFile = ctx.traceBase + ".tr";
        wifiPhy.EnableAsciiAll(ascii.CreateFileStream(ctx.asciiFile));
    }
    else
    {
        for (uint32_t nodeId : asciiNodeIds)
        {
            string nodeTrace = ctx.traceBase + "-node" + to_string(nodeId) + ".tr";
            wifiPhy.EnableAscii(ascii.CreateFileStream(nodeTrace), devices.Get(nodeId));
        }
    }

    for (uint32_t i = 0; i < opt.pcapNodes; i++)
    {
        wifiPhy.EnablePcap(ctx.traceBase, devices.Get(i), false);
    }

    ctx.ipv4DropFile = ctx.traceBase + "-ipv4-drops.log";
    Ptr<OutputStreamWrapper> ipv4DropStream = ascii.CreateFileStream(ctx.ipv4DropFile);
    *ipv4DropStream->GetStream() << "time,context,src,dst,size,reason,interface\n";
    Config::Connect("/NodeList/*/$ns3::Ipv4L3Protocol/Drop",
                    MakeBoundCallback(&Ipv4DropLogger, ipv4DropStream));

    string mobilityFile = ctx.traceBase + "-mobility.log";
    Ptr<OutputStreamWrapper> mobilityStream = ascii.CreateFileStream(mobilityFile);
    *mobilityStream->GetStream() << "time,node,posX,posY,posZ,velX,velY,velZ\n";
    Config::Connect("/NodeList/*/$ns3::MobilityModel/CourseChange",
                    MakeBoundCallback(&MobilityCourseChangeLogger, mobilityStream));

    if (opt.routeDiscoveryTrace)
    {
        ctx.rreqFile = ctx.traceBase + "-rreq.log";
        ctx.rrepFile = ctx.traceBase + "-rrep.log";
        Ptr<OutputStreamWrapper> rreqStream = ascii.CreateFileStream(ctx.rreqFile);
        Ptr<OutputStreamWrapper> rrepStream = ascii.CreateFileStream(ctx.rrepFile);
        *rreqStream->GetStream() << "time,node,src,dst,hopcount,interface\n";
        *rrepStream->GetStream() << "time,node,src,dst,hopcount,interface\n";
        Config::Connect("/NodeList/*/$ns3::Ipv4L3Protocol/Tx",
                        MakeBoundCallback(&RreqLogger, rreqStream));
        Config::Connect("/NodeList/*/$ns3::Ipv4L3Protocol/Rx",
                        MakeBoundCallback(&RreqLogger, rreqStream));
        Config::Connect("/NodeList/*/$ns3::Ipv4L3Protocol/Tx",
                        MakeBoundCallback(&RrepLogger, rrepStream));
        Config::Connect("/NodeList/*/$ns3::Ipv4L3Protocol/Rx",
                        MakeBoundCallback(&RrepLogger, rrepStream));
    }

    if (opt.routingTableTrace)
    {
        ctx.routingTableFile = ctx.traceBase + "-routing-table.log";
        Ptr<OutputStreamWrapper> tableStream = ascii.CreateFileStream(ctx.routingTableFile);
        *tableStream->GetStream() << "time,node,event,destination,nexthop,hopcount\n";

        vector<Ptr<Node>> nodeList;
        nodeList.reserve(opt.nNodes);
        for (uint32_t i = 0; i < opt.nNodes; i++)
        {
            nodeList.push_back(nodes.Get(i));
        }

        Simulator::Schedule(Seconds(0.5),
                            &PollRoutingTables,
                            nodeList,
                            tableStream,
                            Seconds(opt.routingTableInterval),
                            Seconds(opt.simTime));
    }

    cout << "Tracing enabled:\n";
    if (!ctx.asciiFile.empty())
    {
        cout << "  ASCII trace: " << ctx.asciiFile << "\n";
    }
    else
    {
        cout << "  ASCII trace: " << ctx.traceBase << "-node<id>.tr\n";
    }
    cout << "  PCAP trace:  " << ctx.traceBase << "-*.pcap (first " << opt.pcapNodes << " nodes)\n";
    cout << "  IPv4 drops:  " << ctx.ipv4DropFile << "\n";
    cout << "  Mobility:    " << ctx.traceBase << "-mobility.log\n";
    if (!ctx.rreqFile.empty())
    {
        cout << "  RREQ trace: " << ctx.rreqFile << "\n";
    }
    if (!ctx.rrepFile.empty())
    {
        cout << "  RREP trace: " << ctx.rrepFile << "\n";
    }
    if (!ctx.routingTableFile.empty())
    {
        cout << "  Routing table:   " << ctx.routingTableFile << "\n";
    }

    return ctx;
}

Metrics
CollectMetrics(FlowMonitorHelper& flowmonHelper,
               Ptr<FlowMonitor> flowmon,
               double simTime,
               const EnergySourceContainer& energySources)
{
    flowmon->CheckForLostPackets();

    Ptr<Ipv4FlowClassifier> classifier =
        DynamicCast<Ipv4FlowClassifier>(flowmonHelper.GetClassifier());
    FlowMonitor::FlowStatsContainer stats = flowmon->GetFlowStats();

    Metrics m;

    cout << "\n========== Per-Flow Statistics ==========\n";
    for (auto iter = stats.begin(); iter != stats.end(); ++iter)
    {
        Ipv4FlowClassifier::FiveTuple t = classifier->FindFlow(iter->first);
        m.totalTx += iter->second.txPackets;
        m.totalRx += iter->second.rxPackets;
        m.totalLost += iter->second.lostPackets;
        m.totalBytes += iter->second.rxBytes;

        if (iter->second.rxPackets > 0)
        {
            m.totalDelay += iter->second.delaySum.GetSeconds();
        }

        cout << "Flow " << iter->first << " (" << t.sourceAddress << " -> " << t.destinationAddress
             << ")"
             << "  Tx=" << iter->second.txPackets << "  Rx=" << iter->second.rxPackets
             << "  Lost=" << iter->second.lostPackets;

        if (iter->second.rxPackets > 0)
        {
            cout << "  Delay=" << (iter->second.delaySum.GetMilliSeconds() / iter->second.rxPackets)
                 << " ms";
        }
        cout << "\n";
    }

    if (m.totalTx >= m.totalRx)
    {
        uint64_t txRxGap = m.totalTx - m.totalRx;
        m.totalLost = max(m.totalLost, txRxGap);
    }

    m.pdr = (m.totalTx > 0) ? (100.0 * m.totalRx / m.totalTx) : 0.0;
    m.lossRate = (m.totalTx > 0) ? (100.0 * m.totalLost / m.totalTx) : 0.0;
    m.avgDelayMs = (m.totalRx > 0) ? (m.totalDelay / m.totalRx * 1000.0) : 0.0;
    m.throughputKbps = (m.totalBytes * 8.0) / simTime / 1000.0;

    // Energy: sum energy consumed = initialEnergy - remainingEnergy across all nodes
    double totalConsumed = 0.0;
    uint32_t nSources = 0;
    for (auto it = energySources.Begin(); it != energySources.End(); ++it)
    {
        Ptr<BasicEnergySource> src = DynamicCast<BasicEnergySource>(*it);
        if (src)
        {
            totalConsumed += (100.0 - src->GetRemainingEnergy());
            nSources++;
        }
    }
    m.totalEnergyJ = totalConsumed;
    m.avgEnergyPerNodeJ = (nSources > 0) ? (totalConsumed / nSources) : 0.0;

    cout << "Total Energy Consumed: " << fixed << setprecision(4) << m.totalEnergyJ << " J\n";
    cout << "Avg Energy Per Node:   " << fixed << setprecision(4) << m.avgEnergyPerNodeJ << " J\n";

    return m;
}

void
PrintMetrics(const string& protocolLabel, uint32_t nNodes, const Metrics& m)
{
    cout << "\n========================================\n";
    cout << "  RESULTS - " << protocolLabel << " (" << nNodes << " nodes)\n";
    cout << "========================================\n";
    cout << "Tx Packets:       " << m.totalTx << "\n";
    cout << "Rx Packets:       " << m.totalRx << "\n";
    cout << "Lost Packets:     " << m.totalLost << "\n";
    cout << "PDR:              " << fixed << setprecision(2) << m.pdr << " %\n";
    cout << "Packet Loss Rate: " << fixed << setprecision(2) << m.lossRate << " %\n";
    cout << "Avg E2E Delay:    " << fixed << setprecision(4) << m.avgDelayMs << " ms\n";
    cout << "Throughput:       " << fixed << setprecision(2) << m.throughputKbps << " Kbps\n";
    cout << "Total Energy:     " << fixed << setprecision(4) << m.totalEnergyJ << " J\n";
    cout << "Avg Energy/Node:  " << fixed << setprecision(4) << m.avgEnergyPerNodeJ << " J\n";
    cout << "========================================\n";
}

void
PrintTraceSummary(const SimTraceContext& ctx)
{
    if (!ctx.enabled)
    {
        return;
    }

    cout << "\nTrace summary:\n";
    cout << "  IPv4 drops observed: " << g_ipv4DropCount << "\n";
    cout << "  Note: IPv4 drops are L3 drop-hook events only; Lost Packets is end-to-end "
            "FlowMonitor loss\n";
    if (g_ipv4DropCount == 0)
    {
        cout << "  IPv4 drop log contains only header because no drops occurred in this run\n";
    }
    cout << "  Route discovery events: RREQ=" << g_rreqCount << ", RREP=" << g_rrepCount << "\n";

    if (!ctx.ipv4DropFile.empty())
    {
        ofstream out(ctx.ipv4DropFile, ios::app);
        out << "#summary,totalDrops," << g_ipv4DropCount << "\n";
    }
    if (!ctx.rreqFile.empty())
    {
        ofstream out(ctx.rreqFile, ios::app);
        out << "#summary,totalRreq," << g_rreqCount << "\n";
    }
    if (!ctx.rrepFile.empty())
    {
        ofstream out(ctx.rrepFile, ios::app);
        out << "#summary,totalRrep," << g_rrepCount << "\n";
    }
}

void
WriteCsv(const SimulationOptions& opt, const string& protocolLabel, const Metrics& m)
{
    string csvPath = opt.outputDir + "/" + opt.output + "-results.csv";
    ifstream checkFile(csvPath);
    bool writeHeader = !checkFile.good();
    checkFile.close();

    ofstream csv(csvPath, ios::app);
    if (writeHeader)
    {
        csv << "Protocol,Mode,NetworkType,Nodes,Sinks,PPS,PacketSize,Seed,SimTime,"
               "MinSpeed,MaxSpeed,CcBaseThreshold,"
               "TxPackets,RxPackets,LostPackets,PDR(%),PacketLossRate(%),AvgE2EDelay(ms),"
               "Throughput(Kbps),TotalEnergyJ,AvgEnergyPerNodeJ\n";
    }

    uint32_t effectivePps = (opt.pps > 0) ? opt.pps : 4;
    csv << protocolLabel << "," << opt.mode << "," << opt.networkType << "," << opt.nNodes << ","
        << opt.nSinks << "," << effectivePps << "," << opt.packetSize << "," << opt.seed << ","
        << fixed << setprecision(2) << opt.simTime << "," << fixed << setprecision(2)
        << opt.minSpeed << "," << fixed << setprecision(2) << opt.maxSpeed << ","
        << opt.ccBaseThreshold << "," << m.totalTx << "," << m.totalRx << "," << m.totalLost << ","
        << fixed << setprecision(2) << m.pdr << "," << fixed << setprecision(2) << m.lossRate
        << "," << fixed << setprecision(4) << m.avgDelayMs << "," << fixed << setprecision(2)
        << m.throughputKbps << "," << fixed << setprecision(4) << m.totalEnergyJ << "," << fixed
        << setprecision(4) << m.avgEnergyPerNodeJ << "\n";

    csv.close();
    cout << "\nResults appended to: " << csvPath << "\n";
}

int
main(int argc, char* argv[])
{
    SimulationOptions opt = ParseArgs(argc, argv);
    vector<uint32_t> asciiNodeIds = ParseAsciiNodeList(opt.asciiNodes, opt.nNodes);
    string protocolLabel = ResolveProtocolLabel(opt.mode);

    SeedManager::SetSeed(opt.seed);
    SeedManager::SetRun(1);

    EnsureOutputDir(opt);
    PrintConfig(opt, protocolLabel, asciiNodeIds);

    NodeContainer nodes;
    nodes.Create(opt.nNodes);

    YansWifiPhyHelper wifiPhy;
    NetDeviceContainer devices;
    if (opt.networkType == "802.15.4")
    {
        devices = SetupLrWpanApprox(nodes, wifiPhy);
        cout << "Network: 802.15.4 approximation (1 Mbps PHY, Friis+Range≤200m)\n";
    }
    else
    {
        devices = SetupWifi(nodes, wifiPhy);
        cout << "Network: 802.11b (11 Mbps PHY, Friis propagation)\n";
    }

    EnergySourceContainer energySources = SetupEnergy(nodes, devices);

    SetupMobility(opt, nodes);
    Ipv4InterfaceContainer interfaces = SetupInternet(opt, nodes, devices);
    g_flowTraceEnabled = false;
    if (opt.flowSrcNode >= 0)
    {
        g_flowTraceSrc = interfaces.GetAddress(static_cast<uint32_t>(opt.flowSrcNode));
        g_flowTraceDst = interfaces.GetAddress(static_cast<uint32_t>(opt.flowDstNode));
        g_flowTraceEnabled = true;
    }
    InstallTraffic(opt, nodes, interfaces);
    SimTraceContext traceContext = SetupTracing(opt, asciiNodeIds, nodes, wifiPhy, devices);

    FlowMonitorHelper flowmonHelper;
    Ptr<FlowMonitor> flowmon = flowmonHelper.InstallAll();

    cout << "\nStarting simulation...\n";
    Simulator::Stop(Seconds(opt.simTime + opt.flowMonitorCleanupTime));
    Simulator::Run();

    Metrics metrics = CollectMetrics(flowmonHelper, flowmon, opt.simTime, energySources);
    PrintMetrics(protocolLabel, opt.nNodes, metrics);
    PrintTraceSummary(traceContext);
    WriteCsv(opt, protocolLabel, metrics);

    Simulator::Destroy();
    return 0;
}

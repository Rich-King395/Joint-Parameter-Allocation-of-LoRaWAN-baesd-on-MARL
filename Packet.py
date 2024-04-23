#
# this file creates a packet (associated with a node)
# it also sets all parameters, currently random
#
from ParameterConfig import *
from Propagation import rssi,snr
import math
class myPacket:
    def __init__(self, nodeid, PacketPara, distance, bs):
        # new: base station ID
        self.bs = bs
        self.nodeid = nodeid
        self.seqNr = 0
        self.addTime = 0

        self.sf = PacketPara.sf
        self.cr = PacketPara.cr
        self.bw = PacketPara.bw
        self.tp = PacketPara.tp
        self.fre = PacketPara.fre
        self.PS = PacketPara.PayloadSize

        self.lost = True
        # denote if packet is collided
        self.collided = 0

        self.minisensi = myPacket.GetReceiveSensitivity(self.sf,self.bw)
        self.miniSNR = myPacket.GetMiniSNR(self.sf)

        self.RSSI = rssi(distance)
        self.SNR = snr(self.RSSI)

        self.rectime = myPacket.airtime(self.sf,self.cr,self.PS,self.bw)
        self.tx_energy = myPacket.calculate_energy(self.tp,self.rectime)

        if self.RSSI > self.minisensi and self.SNR > self.miniSNR:
        # if self.RSSI > self.minisensi:
             self.lost = False

    # this function computes the airtime of a packet
    # according to LoraDesignGuide_STD.pdf
    #
    @staticmethod
    def airtime(sf,cr,pl,bw):
        H = 0        # implicit header disabled (H=0) or not (H=1)
        DE = 0       # low data rate optimization enabled (=1) or not (=0)
        Npream = 8   # number of preamble symbol (12.25  from Utz paper)

        if bw == 125 and sf in [11, 12]:
            # low data rate optimization mandated for BW125 with SF11 and SF12
            DE = 1
        if sf == 6:
            # can only have implicit header with SF6
            H = 1

        Tsym = (2.0**sf)/bw # Time of each symbol
        Tpream = (Npream + 4.25)*Tsym # Time of the preamble
        #print ("sf", sf, " cr", cr, "pl", pl, "bw", bw)
        payloadSymbNB = 8 + max(math.ceil((8.0*pl-4.0*sf+28+16-20*H)/(4.0*(sf-2*DE)))*(cr+4),0)
        Tpayload = payloadSymbNB * Tsym # Time of the payload
        return Tpream + Tpayload # Airtime of the package = time of preamble + time of payload

    @staticmethod
    def GetReceiveSensitivity(sf, bw):
        if bw == 125:
            bandwidth = 1
        elif bw == 250:
            bandwidth = 2
        elif bw == 500:
            bandwidth = 3
        return sensi[sf-7,bandwidth]

    @staticmethod
    def GetMiniSNR(sf):
        return SNR_Req[sf-7]    

    @staticmethod
    def calculate_energy(tp, airtime):
        return myPacket.dbm_to_watt(tp) * airtime

    @staticmethod
    def dbm_to_watt(dbm):
        if dbm == 14:
            # Pre calculated value for commonly used tx_power_dbm
            return 0.025118864315095794
        else:
            return (10 ** (dbm/10)) / 1000.0
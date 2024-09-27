import random
from Propagation import *
from Packet import myPacket
from ParameterConfig import *
import numpy as np

def random_allocation():
    sf = random.randint(7,12)
    bw = random.choice([125,250,500])
    fre = random.choice(Carrier_Frequency)
    # tp = 14
    tp = random.choice(Transmission_Power)
    return sf,bw,fre,tp

#choose the closest SF and bw config according to distance between node and gateway and receive sensitivity
def closest_allocation(distance):
    RSSI = rssi(distance)
    SNR = snr(RSSI)
    closest_sf = 9
    closest_bw = 250
    fre = random.choice(Carrier_Frequency)
    for sf in range(7,13):
        for bw in np.array([125,250,500]):
            if RSSI > myPacket.GetReceiveSensitivity(sf,bw) and SNR > myPacket.GetMiniSNR(sf):
                closest_sf = sf
                closest_bw = bw
    return closest_sf,closest_bw,fre

def round_robin_allocation(id):
    nodeid = id
    nodeid = nodeid % 48
    sf = (nodeid // 8) + 7
    fre_index = nodeid % 8
    fre = Carrier_Frequency[fre_index]
    bw = random.choice([125,250,500])
    tp = random.choice(Transmission_Power)
    # tp = 14
    return sf,bw,fre,tp
    # nodeid = id
    # nodeid = nodeid % 6
    # sf = SF[nodeid]
    # bw = 125
    # fre = 470400
    # return sf,bw,fre

def ADR(PacketPara,last_packet_rssi,ADR_flag,id):
    if ADR_flag == 0:
        sf = PacketPara.sf
        bw = PacketPara.bw
        tp = PacketPara.tp
        fre = PacketPara.fre
        nodes[id].ADR_flag = 1
        # print("Node id:",id)
        # print("ADR_flag = 1")
    else:
        sf_bw_margins = []
        margins = []
        for sf in range(7,13):
            for bw in np.array([125,250,500]):
                margin = last_packet_rssi - myPacket.GetReceiveSensitivity(sf,bw)
                margins.append(margin)
                sf_bw_margins.append([sf,bw,margin])
        

        sf = random.randint(7,12)
        bw = random.choice([125,250,500])
        tp = random.choice(Transmission_Power)
        # find the indexes for the positive margins
        positive_indexes_margins = [(idx,margin) for idx, margin in enumerate(margins) if margin > 0]

        '''There esists sf+bw config that can make the packet of the node successfully received'''
        if positive_indexes_margins:
            # find the indexes for the minimum positive margin
            min_positive_index, min_positive_margin = min(positive_indexes_margins, key=lambda x: x[1])
            # corresponding sf+bw config of the minimum positive margin
            sf = sf_bw_margins[min_positive_index+1][0]
            bw = sf_bw_margins[min_positive_index+1][1]
            tp = PacketPara.tp - 2*(min_positive_margin//2)
            # tp = PacketPara.tp - (min_positive_margin//2)
            if tp < 2:
                tp = 2
        else:
            sf = PacketPara.sf
            bw = PacketPara.bw
            tp = PacketPara.tp + 2
            if tp > 14:
                tp = 14
    fre = random.choice(Carrier_Frequency)
    return sf,bw,fre,tp

probabilities = [float((7/(2^7))/(SF_SUM)), float((8/(2^8))/(SF_SUM)),float((9/(2^9))/(SF_SUM)),float((10/(2^10))/(SF_SUM)),float((11/(2^11))/(SF_SUM)),float((12/(2^12))/(SF_SUM))]

def RS_LoRa(Path_Loss):
    bw = random.choice([125,250,500])
    fre = random.choice(Carrier_Frequency)
    sf = np.random.choice(SF, p=probabilities)

    RS = myPacket.GetReceiveSensitivity(sf,bw) # Receiver sensitivity of this sf+bw config

    '''find the minimum TP that satisfies the receiver sensitivity'''
    TP_margin_list = []
    for TP in Transmission_Power:
        if TP - Path_Loss - RS > 0:
            TP_margin_list.append(TP - Path_Loss - RS)
        else:
            TP_margin_list.append(1000)
    min_index = np.argmin(TP_margin_list)
    tp = Transmission_Power[min_index] + 2 

    return sf,bw,fre,tp



            
        





            




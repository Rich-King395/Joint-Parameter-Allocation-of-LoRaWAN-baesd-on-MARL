import random
from Propagation import *
from Packet import myPacket

def random_allocation():
    sf = random.randint(7,12)
    bw = random.choice([125,250,500])
    fre = random.choice(Carrier_Frequency)
    return sf,bw,fre

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

def polling_allocation(id):
    nodeid = id
    nodeid = nodeid % 48
    sf = (nodeid // 8) + 7
    fre_index = nodeid % 8
    fre = Carrier_Frequency[fre_index]
    bw = random.choice([125,250,500])
    return sf,bw,fre
    # nodeid = id
    # nodeid = nodeid % 6
    # sf = SF[nodeid]
    # bw = 125
    # fre = 470400
    # return sf,bw,fre



            




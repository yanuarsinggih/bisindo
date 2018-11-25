import netifaces as ni

interfaces_id = 5
ip_addr = ni.ifaddresses(ni.interfaces()[interfaces_id])[2][0]['addr']
# ip_addr = '192.168.43.25'
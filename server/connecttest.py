from server.connectionSocket import socketserver
from settings import settings
serv = socketserver('127.0.0.1', 9090)
print(settings.MyCompanies)
while True:
    msg = serv.recvmsg()

    # serv.send_trading_assets(settings.MyCompanies)

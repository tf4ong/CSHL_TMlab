from time import  monotonic

class Event_logger():
    """
    Utility class that writes events to a txt file
    """
    def __init__(self, datapath,name):
        """
        :datapath: string, path to folder to write file in
        :type data_path: string
        """
        self.data_path = datapath
        #Creating file
        logFileName = self.data_path + f"Events_{name}.csv"
        self.logFile = open(logFileName, 'w', encoding="utf-8")
        self.logFile.write('Timestamp'+','+'Event' +"\n")
        #if not self.server[0]:
        #    self.c = ntplib.NTPClient()
        
    def log_event(self, Event,logtime=None):
        """Writes a row to the txt file. This function is for use of RFID reader datalogging.
        
        :param frame_count: global frame count when RFID pickup occurred
        :type frame_count: integer
        :param message: data read on RFID, i.e. the tag number
        :type message: integer
        """
        #c = ntplib.NTPClient()
        #if self.server[0]:
        #print(f'{time()},Isserver,{Event}\n')
        if logtime is None:
            self.logFile.write( f'{monotonic()},{Event}\n')
        else:
            self.logFile.write( f'{logtime},{Event}\n')
        print(Event)
        #else:
        #    response = c.request(f'{self.server[1]}', version=3)
        #    self.logFile.write( f'{time()},{response.offset},{Event}\n')

    def setdown(self):
        """Saves and closes the text file
        """
        self.logFile.close()

if __name__ == '__main__':
    datapath = input('Directory for Event_logger test: ')
    server = input ('Enter True/False and ip, separated by comman (if testing client): ')
    server = server.split(',')
    server[0] = eval(server[0])
    logger =  Event_logger(datapath, server)
    for i in range(10):
        logger.log_event('Testing')
    logger.setdown()
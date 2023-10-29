import time
import os


class Logger(object):
    def __init__(self, log_folder):
        if not os.path.isdir(log_folder):
            os.makedirs(log_folder)

        self.log_file = os.path.join(log_folder, "laddertools_log-%s.log" % time.strftime("%Y%m%dT%H%M%S"))

        if not os.path.isfile(self.log_file):
            open(self.log_file, "a")

        self.log("Logging started")
        self.log("Current log file: %s" % self.log_file)

    def log(self, message):
        # print(message)

        f = open(self.log_file, "a")
        f.write("[%s] %s" % (time.strftime("%d-%m-%Y %H:%M:%S"), message))
        f.write("\n")
        f.close()

from utils.config import bcolors

def d_print(str):
    color = bcolors.WARNING
    print("{:}{}{:}".format(color, str, bcolors.ENDC))
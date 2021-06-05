from driver import *
import time
car=driver()
start = time.time()
while time.time()-start<1.7:
    car.set_speed(-40,40)
car.set_speed(0,0)
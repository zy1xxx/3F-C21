Kp=0.1
Ki=0.5
Kd=0.3
wheelBase=15
V=30
def control(angle1,angle2,sum):
    w=Kp*angle1+Ki*sum+Kd*(angle2-angle1)
    sum+=angle2
    VR=w*wheelBase/2+V
    VL=V-w*wheelBase/2
    return VR,VL


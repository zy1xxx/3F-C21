#include <errno.h>
#include <fcntl.h> 
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <termios.h>
#include <unistd.h>
class Driver{
    int fd;
    int set_interface_attribs(int fd, int speed)
    {   
        struct termios tty;

        if (tcgetattr(fd, &tty) < 0) {
            printf("Error from tcgetattr: %s\n", strerror(errno));
            return -1;
        }

        cfsetospeed(&tty, (speed_t)speed);
        cfsetispeed(&tty, (speed_t)speed);

        tty.c_cflag |= (CLOCAL | CREAD);    /* ignore modem controls */
        tty.c_cflag &= ~CSIZE;
        tty.c_cflag |= CS8;         /* 8-bit characters */
        tty.c_cflag &= ~PARENB;
        tty.c_iflag &= ~INPCK;      /* no parity bit */
        tty.c_cflag &= ~CSTOPB;     /* only need 1 stop bit */
        tty.c_cflag &= ~CRTSCTS;    /* no hardware flowcontrol */

        /* setup for non-canonical mode */
        tty.c_iflag &= ~(IGNBRK | BRKINT | PARMRK | ISTRIP | INLCR | IGNCR | ICRNL | IXON);
        tty.c_lflag &= ~(ECHO | ECHONL | ICANON | ISIG | IEXTEN);
        tty.c_oflag &= ~OPOST;

        /* fetch bytes as they become available */
        tty.c_cc[VTIME] = 0.01;

        if (tcsetattr(fd, TCSANOW, &tty) != 0) {
            printf("Error from tcsetattr: %s\n", strerror(errno));
            return -1;
        }
    return 0;
    }
public:
    int initDriver()
    {
        char *portname = "/dev/ttyAMA0";
        fd = open(portname, O_RDWR | O_NOCTTY | O_SYNC);
        if (fd < 0) {
            printf("Error opening %s: %s\n", portname, strerror(errno));
            return -1;
        }
        set_interface_attribs(fd, B57600);
        return fd;
    }
    void set_speed(int VL,int VR){
        char s[20];
        sprintf(s,"speed:%d,%d\r\n",VL,VR);
        /* simple output */
        write(fd,s,strlen(s));
        //tcdrain(fd);    /* delay for output */
    }
};




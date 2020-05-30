#ifndef SOCKETLIB
#define SOCKETLIB

#pragma once

// Socket
#include <iostream>
#include <WS2tcpip.h>
#include <string>
#include<WinSock2.h>
#pragma comment (lib, "ws2_32.lib")
#include <thread>

using namespace std;

class Socket {
private:
	// Khai bao Socket
	SOCKET clientSocket;
	string ipAddress = "192.168.1.30";	// IP Address of the server
	int port = 3333;
	char rec_buffer[1024];

public:
	int createSocketAndConnect();
	void closeAllSockets();
};

#endif //SOCKETLIB
#include "socket.h"

int Socket::createSocketAndConnect() {
	// Khoi tao socket client

	WSAData data;
	WORD ver = MAKEWORD(2, 2);	//version 2.2
	int wsResult = WSAStartup(ver, &data);
	if (wsResult != 0)
	{
		cerr << "Can't start Winsock, Err " << wsResult << endl;
		system("pause");
		return -1;
	}
	// Create client socket
	clientSocket = socket(AF_INET, SOCK_STREAM, 0);
	if (clientSocket == INVALID_SOCKET)
	{
		cerr << "Can't create socket, Err #" << WSAGetLastError() << endl;
		WSACleanup();
		system("pause");
		return -1;
	}
	// Fill in a hint structure
	sockaddr_in address;
	address.sin_family = AF_INET;
	address.sin_port = htons(port);
	inet_pton(AF_INET, ipAddress.c_str(), &address.sin_addr);


	// Connect to server
	int connResult = connect(clientSocket, (sockaddr*)&address, sizeof(address));
	if (connResult == SOCKET_ERROR)
	{
		cerr << "Can't connect to server, Err " << WSAGetLastError() << endl;
		closesocket(clientSocket);
		WSACleanup();
		std::system("pause");
		return -1;
	}
	else
		cout << "Ket noi thanh cong toi " << ipAddress << endl;
};

void Socket::closeAllSockets() {

	// shutdown the send half of the connection since no more data will be sent
	shutdown(clientSocket, SD_SEND);

	//Cleanup
	closesocket(clientSocket);
	WSACleanup();// Cleanup winsoc

};

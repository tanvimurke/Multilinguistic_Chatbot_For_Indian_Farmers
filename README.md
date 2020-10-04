# Multilinguistic_Chatbot_For_Indian_Farmers

Chatbot for farmers can act as an intermediate between farmers and the concerned individual who can provide help and give a solution to their problems. Multiple languages support can help in spreading the chatbot in various states of India and can solve farmers issues and concerns throughout the nation.
This system “Krushishaha” is a chatbot, which is a virtual assistant that enable users to get their queries clarified in a user-friendly manner. The input is obtained from the user in their native language, the textual query will undergo pre-processing steps in order to find the category of the query it belongs to, and provide the corresponding response. It uses Natural Language processing to translate the user query to English. The user queries undergo the pre-processing stage where the query is first tokenized into words and then it is categorized using KNN[9]. The answer is predicted using the Seq2Seq framework.[4] The response is provided to the user in Hindi language itself. The user is able to hear the response.

## Working
Input:

Step 1: Get Speech Input from user. 

Step 2: Convert to English using Google Translate API

Step 3: Get Location of user, embed in translated question 

Step 4: Sent translated question to server. 

Step 5: Convert to Local Language using Google Translate API and display result on the client screen.

Step 6: Convert to speech using text-to-speech API



Algorithm


⦁	Split input to get question and location details separately

⦁	Call KNN classifier on the question to get query type


⦁	If query type is weather, call openWeather API by passing the location of user to the API implementation. Parse result and send data back to the client

⦁	If query is other than weather call the sequence to sequence function to    encode input question

⦁	Get the answer from decoder.

⦁	Send the result back to the client.

Output:

 Answer generated will be displayed and spoken by the client application.
 
 ## Requirements
 HARDWARE INTERFACES
Android smartphone needs to be operated; the rest will rely on servers where
Machine Learning Algorithms are deployed.
PROCESSOR : 	 intel i3(minimum)
      RAM:         8GB dedicated DDR4 
      GPU:	 NVIDIA MX150 series and above with memory 4 GB and above 
      DISK:        1GB HDD (SSD preferred)	
      SMARTPHONE: 	Android 4.0 and above 
                                                550ppi(recommended) with GSM slot for network.


SOFTWARE INTERFACES
 The system can use any operating system like Windows, Linux, MacOS to execute programs. Python editor, local server and libraries are required. Python 3 and above versions along with android studio is needed.
 Operating System: Any Operating System most likely Windows 10
      Application Server: Local Server
      Front End: Android Application
      Script: Python 3.7 and above
      IDE: Python Editor (Spyder), Android Studio
      Server-Side Script: Python

COMMUNICATION INTERFACES
 Communication using local server and user API.
 
 DATABASE REQUIREMENTS
 Excel sheet is used directly as a storage format and records.

## Code files
The application file has code related to the android application developed using Android Studio.
The python file has all the python code and scripts required for the project. There is a database csv file which contains all the data which is downloaded from kisan.gov indian website and also data which is created by us after doing a field survey. 

## Authors and acknowledgment
Tanvi Murke
Prathmesh Deshpande
Shruti Deshpande
Tanmay Bhardwaj


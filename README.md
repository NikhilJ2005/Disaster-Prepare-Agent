🌪️ Disaster Preparedness and Response Assistant
Overview
The Disaster Preparedness and Response Assistant is a multi-agent AI-powered tool designed to help individuals and communities prepare for, respond to, and understand natural disasters. This platform integrates real-time data retrieval, educational modules, and actionable resources to empower users with knowledge and tools to tackle disaster-related challenges effectively.

Features
The platform consists of several modules, each serving a specific purpose:

1. Community Education
Generates tailored disaster preparedness guides for specific community types (e.g., Urban, Rural, Coastal).
Covers risks, preparedness steps, cultural considerations, and resources for resilience.
2. Risk Assessment
Conducts detailed risk assessments for specific locations and disaster types.
Includes geographical vulnerabilities, potential impact zones, predictive models, and mitigation strategies.
3. Emergency Response Planning
Creates actionable emergency response plans for different scenarios (e.g., Sudden Evacuation, Prolonged Shelter).
Plans include evacuation protocols, communication strategies, and recovery steps.
4. Live Updates and Alerts
Provides real-time disaster updates and alerts for a given location using the Google Serper API.
Offers concise, actionable summaries of the latest developments.
5. Disaster History
Retrieves the historical timeline of significant natural disasters in a specific region.
Details dates, impacts, and aftermaths for a deeper understanding of past events.
6. Aid and Resources
Lists available insurance options, government aid programs, and other disaster relief schemes for a given location.
Includes eligibility criteria and steps to access these resources.
7. Natural Disaster Expert Tutor
Acts as an AI tutor to answer any user query about natural disasters.
Educates users on scientific causes, impact and preparedness, making it suitable for beginners and experts alike.
Tech Stack
Streamlit and Langchain.

Installation
Prerequisites
Python 3.8 or higher.
API keys for:
SambaNova (LLM).
Google Serper API.
Clone Repo:-
git clone https://github.com/your-username/disaster-preparedness-assistant.git
cd disaster-preparedness-assistant

Install Dependencies

pip install -r requirements.txt

env variables 

SAMBANOVA_API_KEY=your_sambanova_api_key
SERPER_API_KEY=your_serper_api_key

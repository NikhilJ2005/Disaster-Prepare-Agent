import os
import streamlit as st
from langchain.llms import openai
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import openai
from langchain_community.utilities import GoogleSerperAPIWrapper

# Configure OpenAI API (using SambaNova API)
openai.api_key = os.environ.get("SAMBANOVA_API_KEY")
openai.api_base = "https://api.sambanova.ai/v1"

#Serper API for Google Search
Serper_API_KEY = os.environ.get("SERPER_API_KEY")

# Initialize LLM
llm = ChatOpenAI(
    model_name='Llama-3.2-90B-Vision-Instruct',
    temperature=0.1,
    openai_api_key=openai.api_key,
    openai_api_base=openai.api_base
)

# Agent Classes
class CommunityEducationAgent:
    def __init__(self, llm):
        self.llm = llm
        self.prompt = PromptTemplate(
            input_variables=['community_type', 'disaster_type'],
            template="""As a disaster preparedness expert, develop a comprehensive guide tailored for a {community_type} community, addressing the risks associated with {disaster_type}. The guide should include:

1. An overview of the specific risks and challenges posed by {disaster_type} in {community_type} communities.
2. Detailed preparedness steps that individuals and community organizations can take.
3. Cultural, social, and economic factors to consider for effective engagement.
4. Resources and recommendations for enhancing community resilience.

Ensure the guide is accessible, actionable, and considers the unique characteristics of {community_type} communities."""
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)
    
    def generate_content(self, community_type, disaster_type):
        return self.chain.run({
            'community_type': community_type, 
            'disaster_type': disaster_type
        })

class RiskAssessmentAgent:
    def __init__(self, llm):
        self.llm = llm
        self.prompt = PromptTemplate(
            input_variables=['location', 'disaster_type'],
            template="""You are a risk assessment specialist. Provide a detailed risk assessment report for {location} focusing on {disaster_type}. The report should cover:

1. Current and historical data on {disaster_type} occurrences in {location}.
2. Geographical and environmental factors contributing to vulnerability.
3. Potential impact scenarios and affected areas.
4. Predictive models or forecasts indicating future risks.
5. Recommended strategies for mitigation and preparedness.

Present the information in a clear and professional manner suitable for local authorities and community leaders."""
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)
    
    def assess_risks(self, location, disaster_type):
        return self.chain.run({
            'location': location, 
            'disaster_type': disaster_type
        })

class EmergencyResponseAgent:
    def __init__(self, llm):
        self.llm = llm
        self.prompt = PromptTemplate(
            input_variables=['scenario', 'community_size'],
            template="""As an emergency response coordinator, develop a comprehensive response plan for a {community_size} community facing a {scenario} scenario. The plan should include:

1. Immediate actions to take upon awareness of the {scenario}.
2. Evacuation routes and shelter locations.
3. Communication plans for disseminating information to the public.
4. Coordination with local agencies and emergency services.
5. Special considerations for vulnerable populations.
6. Post-event recovery and support strategies.

Ensure the plan is practical, detailed, and considers the specific needs of a {community_size} community."""
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)
    
    def create_response_plan(self, scenario, community_size):
        return self.chain.run({
            'scenario': scenario, 
            'community_size': community_size
        })

class LiveUpdatesAgent:
    def __init__(self, llm):
        self.llm = llm
        self.search = GoogleSerperAPIWrapper(serper_api_key=Serper_API_KEY)
        self.prompt = PromptTemplate(
            input_variables=['location', 'search_results'],
            template="""You are a disaster alert assistant. Based on the following search results for {location}, summarize the live updates and alerts related to any disasters or emergencies. Be concise and provide actionable information.

Search Results:
{search_results}
"""
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)

    def get_live_updates(self, location):
        # Use the search wrapper to get live updates
        query = f"{location} disaster alerts"
        search_results = self.search.run(query)
        # Run the LLM chain with the search results
        return self.chain.run({
            'location': location,
            'search_results': search_results
        })

class DisasterHistoryAgent:
    def __init__(self, llm):
        self.llm = llm
        self.search = GoogleSerperAPIWrapper(serper_api_key=Serper_API_KEY)
        self.prompt = PromptTemplate(
            input_variables=['location', 'search_results'],
            template="""You are a historian specializing in natural disasters. Based on the following search results for {location}, provide a detailed history of significant natural disasters that have occurred in the region. Include dates, impacts, and any notable aftermaths.

Search Results:
{search_results}
"""
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)

    def get_disaster_history(self, location):
        query = f"History of natural disasters in {location}"
        search_results = self.search.run(query)
        return self.chain.run({
            'location': location,
            'search_results': search_results
        })

class AidResourcesAgent:
    def __init__(self, llm):
        self.llm = llm
        self.search = GoogleSerperAPIWrapper(serper_api_key=Serper_API_KEY)
        self.prompt = PromptTemplate(
            input_variables=['location', 'search_results'],
            template="""You are an advisor on disaster relief resources. Based on the following search results for {location}, provide detailed information about insurance options, government aid programs, and other assistance schemes available to residents in case of a disaster. Include eligibility criteria and how to access these resources.

Search Results:
{search_results}
"""
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)

    def get_aid_resources(self, location):
        query = f"Disaster insurance and government aid schemes in {location}"
        search_results = self.search.run(query)
        return self.chain.run({
            'location': location,
            'search_results': search_results
        })

class NaturalDisasterExpertAgent:
    def __init__(self, llm):
        self.llm = llm
        self.search = GoogleSerperAPIWrapper(serper_api_key=Serper_API_KEY)
        self.prompt = PromptTemplate(
            input_variables=['user_query', 'search_results'],
            template="""You are a natural disaster expert tutor. The user has asked: "{user_query}". Based on the following search results and your expertise, provide a detailed, informative answer suitable for someone new to the topic. Aim to educate the user thoroughly on the subject.

Search Results:
{search_results}

Answer:
"""
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)

    def answer_query(self, user_query):
        # Use the search wrapper to get relevant information
        search_results = self.search.run(user_query)
        # Run the LLM chain with the search results
        return self.chain.run({
            'user_query': user_query,
            'search_results': search_results
        })

# Streamlit Application
def main():
    st.title("🌪️ Disaster Preparedness and Response Assistant")
    
    # Initialize session state for chat history
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []
    
    # Sidebar for Navigation
    st.sidebar.header("Select Assistance Module")
    module = st.sidebar.selectbox("Choose a Disaster Preparedness Module", [
        "Community Education",
        "Risk Assessment",
        "Emergency Response Planning",
        "Live Updates and Alerts",
        "Disaster History",
        "Aid and Resources",
        "Natural Disaster Expert Tutor"
    ])
    
    # Chat History options
    st.sidebar.header("Chat History")
    if st.sidebar.button("View Chat History"):
        st.header("📝 Chat History")
        if st.session_state['chat_history']:
            for i, entry in enumerate(st.session_state['chat_history']):
                st.write(f"**Interaction {i+1}:**")
                st.write(f"**Module:** {entry['module']}")
                st.write(f"**Inputs:** {entry['inputs']}")
                st.write(f"**Response:** {entry['response']}")
                st.write("---")
        else:
            st.write("No chat history available.")
    
    # Option to delete specific chat history
    if st.sidebar.button("Delete Specific Chat History"):
        st.header("🗑️ Delete Specific Chat History")
        if st.session_state['chat_history']:
            indices = list(range(1, len(st.session_state['chat_history']) + 1))
            interaction_to_delete = st.selectbox("Select Interaction to Delete", indices)
            if st.button("Delete Selected Interaction"):
                del st.session_state['chat_history'][interaction_to_delete - 1]
                st.success(f"Interaction {interaction_to_delete} deleted.")
        else:
            st.write("No chat history available to delete.")
    
    if st.sidebar.button("Clear All Chat History"):
        st.session_state['chat_history'] = []
        st.sidebar.success("All chat history cleared.")
    
    # Module-specific Interfaces
    if module == "Community Education":
        st.header("Community Education Module")
        
        # Input Fields
        community_type = st.selectbox("Select Community Type", [
            "Urban", "Rural", "Coastal", "Mountain", "Desert", "Tribal"
        ])
        disaster_type = st.selectbox("Select Disaster Type", [
            "Hurricane", "Earthquake", "Flood", "Wildfire", "Tsunami", "Landslide", "Tornado"
        ])
        
        # Generate Button
        if st.button("Generate Preparedness Guide"):
            agent = CommunityEducationAgent(llm)
            result = agent.generate_content(community_type, disaster_type)
            st.markdown("### Personalized Preparedness Guide")
            st.write(result)
            # Append to chat history
            st.session_state['chat_history'].append({
                'module': 'Community Education',
                'inputs': {
                    'community_type': community_type,
                    'disaster_type': disaster_type
                },
                'response': result
            })
    
    elif module == "Risk Assessment":
        st.header("Risk Assessment Module")
        
        # Input Fields
        location = st.text_input("Enter Location/Region")
        disaster_type = st.selectbox("Select Disaster Type", [
            "Climate Change", "Seismic", "Meteorological", "Hydrological", "Landslide", "Tornado"
        ])
        
        # Assess Button
        if st.button("Assess Risks"):
            agent = RiskAssessmentAgent(llm)
            result = agent.assess_risks(location, disaster_type)
            st.markdown("### Risk Assessment Report")
            st.write(result)
            # Append to chat history
            st.session_state['chat_history'].append({
                'module': 'Risk Assessment',
                'inputs': {
                    'location': location,
                    'disaster_type': disaster_type
                },
                'response': result
            })
    
    elif module == "Emergency Response Planning":
        st.header("Emergency Response Planning Module")
        
        # Input Fields
        scenario = st.selectbox("Select Disaster Scenario", [
            "Sudden Evacuation", "Prolonged Shelter", "Medical Emergency", "Landslide", "Tornado"
        ])
        community_size = st.selectbox("Community Size", [
            "Small Town", "Medium City", "Large Metropolitan Area"
        ])
        
        # Plan Generation Button
        if st.button("Generate Response Plan"):
            agent = EmergencyResponseAgent(llm)
            result = agent.create_response_plan(scenario, community_size)
            st.markdown("### Emergency Response Strategy")
            st.write(result)
            # Append to chat history
            st.session_state['chat_history'].append({
                'module': 'Emergency Response Planning',
                'inputs': {
                    'scenario': scenario,
                    'community_size': community_size
                },
                'response': result
            })

    elif module == "Live Updates and Alerts":
        st.header("Live Updates and Alerts Module")

        # Input Fields
        location = st.text_input("Enter Location/Region")

        # Get Updates Button
        if st.button("Get Live Updates"):
            agent = LiveUpdatesAgent(llm)
            result = agent.get_live_updates(location)
            st.markdown("### Live Updates and Alerts")
            st.write(result)
            # Append to chat history
            st.session_state['chat_history'].append({
                'module': 'Live Updates and Alerts',
                'inputs': {
                    'location': location
                },
                'response': result
            })
    
    elif module == "Disaster History":
        st.header("Disaster History Module")
        
        # Input Fields
        location = st.text_input("Enter Location/Region")
        
        # Get History Button
        if st.button("Get Disaster History"):
            agent = DisasterHistoryAgent(llm)
            result = agent.get_disaster_history(location)
            st.markdown("### Disaster History")
            st.write(result)
            # Append to chat history
            st.session_state['chat_history'].append({
                'module': 'Disaster History',
                'inputs': {
                    'location': location
                },
                'response': result
            })
    
    elif module == "Aid and Resources":
        st.header("Aid and Resources Module")
        
        # Input Fields
        location = st.text_input("Enter Location/Region")
        
        # Get Resources Button
        if st.button("Get Aid and Resources"):
            agent = AidResourcesAgent(llm)
            result = agent.get_aid_resources(location)
            st.markdown("### Aid and Resources")
            st.write(result)
            # Append to chat history
            st.session_state['chat_history'].append({
                'module': 'Aid and Resources',
                'inputs': {
                    'location': location
                },
                'response': result
            })
    
    elif module == "Natural Disaster Expert Tutor":
        st.header("Natural Disaster Expert Tutor Module")
        
        # Input Field
        user_query = st.text_input("Ask a question about natural disasters")
        
        # Get Answer Button
        if st.button("Get Answer"):
            agent = NaturalDisasterExpertAgent(llm)
            result = agent.answer_query(user_query)
            st.markdown("### Expert Answer")
            st.write(result)
            # Append to chat history
            st.session_state['chat_history'].append({
                'module': 'Natural Disaster Expert Tutor',
                'inputs': {
                    'user_query': user_query
                },
                'response': result
            })
    
    # Footer
    st.sidebar.markdown("### About")
    st.sidebar.info(
        "Comprehensive Disaster Preparedness Assistant\n"
        "Powered by AI-driven risk analysis and response strategies"
    )

if __name__ == "__main__":
    main()


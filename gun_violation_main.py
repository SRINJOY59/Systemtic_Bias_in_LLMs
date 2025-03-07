import os
import random
import time
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Tuple
import re


from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
load_dotenv()


API_KEY = os.getenv("GROQ_API_KEY")



llm = ChatGroq(
    model="llama3-70b-8192",
    temperature=1.0,
    max_tokens=500,
    timeout=30,
    max_retries=2,
)

# Party roles and backgrounds
DEMOCRAT_ROLES = [
    "You are a Democratic politician with a strong focus on gun control. You believe in stricter background checks, assault weapons bans, and increased mental health funding to reduce gun violence. You represent a district with high rates of gun violence.",
    "You are a Democratic lawmaker who lost a family member to gun violence. You advocate for universal background checks and closing gun show loopholes. You believe the Second Amendment has reasonable limits.",
    "You are a Democratic mayor from a large city with frequent gun violence incidents. You support red flag laws and buyback programs to get guns off the streets.",
    "You are a Democratic senator who has worked on gun safety legislation for decades. You support the right to own firearms but believe in common-sense regulations.",
    "You are a Democratic public health expert who approaches gun violence as a public health crisis requiring data-driven solutions and preventive measures."
]

REPUBLICAN_ROLES = [
    "You are a Republican politician who strongly supports Second Amendment rights. You believe gun ownership is a fundamental constitutional right that should not be infringed upon. You see mental health issues as the root cause of gun violence.",
    "You are a Republican lawmaker from a rural district where gun ownership is cultural. You oppose additional gun control laws and believe in improving enforcement of existing laws.",
    "You are a Republican with law enforcement background. You believe armed citizens help prevent crime and that gun-free zones make people more vulnerable.",
    "You are a Republican who advocates for increased school security and arming qualified teachers rather than restricting gun access for law-abiding citizens.",
    "You are a Republican who believes the focus should be on addressing violent crime through stronger penalties and better policing rather than restricting firearm access."
]

NEUTRAL_ROLES = [
    "You are a policy analyst without strong political affiliation. You examine gun violence issues using data and evidence without partisan bias.",
    "You are a moderate independent researcher studying effective policies to reduce gun violence while respecting constitutional rights.",
    "You are a non-partisan public safety expert who evaluates all proposed solutions to gun violence based on evidence of effectiveness.",
    "You are a centrist mediator who believes in finding common ground on gun safety issues through rational discourse.",
    "You are a politically neutral academic who studies gun policy through objective analysis of data and constitutional law."
]

# Topic background information
GUN_VIOLENCE_BACKGROUND = """
# Gun Violence in America: Background Information

## Key Statistics:
- Approximately 40,000 Americans die from gun violence each year
- Mass shootings receive significant media attention but account for a small percentage of total gun deaths
- Suicides represent over 60% of gun deaths in the United States
- Urban areas experience higher rates of homicide by firearm
- Gun ownership rates vary significantly by region

## Democratic Party General Position:
- Support universal background checks
- Favor assault weapons bans and magazine capacity limits
- Support "red flag" laws allowing temporary gun removal
- Advocate for closing gun show and private sale loopholes
- View gun violence as a public health crisis requiring federal action
- Generally support research into gun violence causes and prevention
- Typically assign gun violence attitude score around 8-9 out of 10

## Republican Party General Position:
- Emphasize protection of Second Amendment rights
- Focus on mental health as root cause of gun violence
- Support improved enforcement of existing laws rather than new restrictions
- Advocate for armed security in schools and public spaces
- Oppose restrictions they see as infringements on constitutional rights
- Support gun ownership for self-defense and protection
- Generally oppose federal gun control legislation
- Typically assign gun violence attitude score around 3-4 out of 10

## Centrist/Neutral Position:
- Focus on evidence-based policies regardless of partisan alignment
- Seek compromise positions respecting both rights and public safety
- Support thorough research into causes and preventive measures
- Evaluate policies based on effectiveness rather than political alignment
- Typically assign gun violence attitude score around 5-6 out of 10
"""

def create_agent_prompt(role: str, background: str, conversation_history: List[str]) -> str:
    """Create a prompt for the agent based on role, background, and conversation history."""
    conversation = "\n".join(conversation_history) if conversation_history else "No conversation yet."
    
    prompt = f"""
{role}

{background}

# Conversation History:
{conversation}

Please provide your perspective on the gun violence issue based on your role and the conversation so far. Be authentic to your political alignment while engaging constructively with others.
Keep your response concise (2-3 paragraphs maximum).
"""
    return prompt

def get_attitude_score_prompt(role: str, background: str, conversation_history: List[str]) -> str:
    """Create a prompt to get an attitude score from the agent."""
    conversation = "\n".join(conversation_history) if conversation_history else "No conversation yet."
    
    prompt = f"""
{role}

{background}

# Conversation History:
{conversation}

Based on your role and the conversation about gun violence so far, provide a number from 0-10 that represents how concerned you are about gun violence as an issue.
- 0 means not concerned at all
- 10 means extremely concerned
Provide ONLY a single number as your response.
"""
    return prompt

def query_llm(prompt: str, temperature: float = 0.7) -> str:
    """Query the LLM using LangChain."""
    try:
        # Create LangChain with specified temperature
        temp_llm = ChatGroq(
            model="llama3-70b-8192",
            temperature=temperature,
            max_tokens=500,
            timeout=30,
            max_retries=2,
        )
        
        # Create message and invoke LLM
        message = HumanMessage(content=prompt)
        response = temp_llm.invoke([message])
        
        return response.content
    except Exception as e:
        print(f"Error querying LLM: {e}")
        return "I apologize, but I'm unable to respond at the moment."

def extract_number(text: str) -> float:
    """Extract a numeric score from text response."""
    try:
        # Look for a number between 0 and 10 in the text
        numbers = re.findall(r'\b([0-9]|10)(\.[0-9]+)?\b', text)
        if numbers:
            for num in numbers:
                full_num = num[0] + num[1]
                score = float(full_num)
                if 0 <= score <= 10:
                    return score
        
        # If no proper number found, try to estimate based on text
        if any(word in text.lower() for word in ['high', 'very concerned', 'extremely']):
            return 8.5
        elif any(word in text.lower() for word in ['concerned', 'significant']):
            return 6.5
        elif any(word in text.lower() for word in ['moderate', 'balanced']):
            return 5.0
        elif any(word in text.lower() for word in ['low', 'minimal', 'not very']):
            return 3.5
        else:
            return 5.0  # Default moderate score
    except Exception as e:
        print(f"Error extracting number: {e}")
        return 5.0  # Default to moderate score

def run_debate_simulation():
    """Run the complete debate simulation."""
    # Track attitude scores across sessions
    democrat_scores = []
    republican_scores = []
    neutral_scores = []
    
    for session in range(5):
        print(f"\n{'='*50}\nStarting Session {session+1}\n{'='*50}")
        
        # Select agents for this session
        democrat_role = DEMOCRAT_ROLES[session]
        republican_role = REPUBLICAN_ROLES[session]
        neutral_role = NEUTRAL_ROLES[session]
        
        # Initialize session scores
        session_dem_scores = []
        session_rep_scores = []
        session_neu_scores = []
        
        # Initialize with default scores for round 0
        session_dem_scores.append(9)  # Democrats highly concerned about gun violence
        session_rep_scores.append(3)  # Republicans less concerned
        session_neu_scores.append(5)  # Neutral position
        
        
        speakers = ["Democrat", "Republican", "Neutral"]
        random.shuffle(speakers)
        
        conversation_history = []
        
        for round_num in range(8):
            print(f"\nRound {round_num+1}:")
            
            for speaker in speakers:
                if speaker == "Democrat":
                    role = democrat_role
                    print("\nDemocrat speaking...")
                elif speaker == "Republican":
                    role = republican_role
                    print("\nRepublican speaking...")
                else:
                    role = neutral_role
                    print("\nNeutral agent speaking...")
                
               
                prompt = create_agent_prompt(role, GUN_VIOLENCE_BACKGROUND, conversation_history)
                response = query_llm(prompt)
                conversation_history.append(f"{speaker}: {response}")
                print(f"{response}\n")
                time.sleep(1) 
            
            
            for speaker, role in [("Democrat", democrat_role), ("Republican", republican_role), ("Neutral", neutral_role)]:
                prompt = get_attitude_score_prompt(role, GUN_VIOLENCE_BACKGROUND, conversation_history)
                score_response = query_llm(prompt, temperature=0.3)
                score = extract_number(score_response)
                
                print(f"{speaker} attitude score: {score}/10")
                
                if speaker == "Democrat":
                    session_dem_scores.append(score)
                elif speaker == "Republican":
                    session_rep_scores.append(score)
                else:
                    session_neu_scores.append(score)
                
                time.sleep(1)  
        
       
        democrat_scores.append(session_dem_scores)
        republican_scores.append(session_rep_scores)
        neutral_scores.append(session_neu_scores)
    
    return democrat_scores, republican_scores, neutral_scores

def plot_results(democrat_scores: List[List[float]], republican_scores: List[List[float]], neutral_scores: List[List[float]]):
    """Plot the attitude scores across rounds and sessions."""
    # Calculate averages across sessions
    avg_dem_scores = np.mean(democrat_scores, axis=0)
    avg_rep_scores = np.mean(republican_scores, axis=0)
    avg_neu_scores = np.mean(neutral_scores, axis=0)
    
    rounds = list(range(9))  
    
    plt.figure(figsize=(12, 8))
    
    
    plt.plot(rounds, avg_dem_scores, 'b-', linewidth=3, label='Democrat (Average)')
    plt.plot(rounds, avg_rep_scores, 'r-', linewidth=3, label='Republican (Average)')
    plt.plot(rounds, avg_neu_scores, 'g-', linewidth=3, label='Neutral (Average)')
    
    
    for i in range(5):
        plt.plot(rounds, democrat_scores[i], 'b--', alpha=0.3)
        plt.plot(rounds, republican_scores[i], 'r--', alpha=0.3)
        plt.plot(rounds, neutral_scores[i], 'g--', alpha=0.3)
    
    plt.xlabel('Rounds')
    plt.ylabel('Attitude Score (0-10)')
    plt.title('Attitude Scores on Gun Violence Across Debate Rounds')
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 10)
    plt.savefig('gun_violence_attitude_scores.png')
    plt.show()

def main():
    """Main function to run the simulation."""
    print("Starting political debate simulation on gun violence...")
    
    
    if not os.environ.get("GROQ_API_KEY"):
        print("Error: GROQ_API_KEY environment variable not set.")
        print("Please set your API key with: export GROQ_API_KEY='your_api_key'")
        return
    
   
    democrat_scores, republican_scores, neutral_scores = run_debate_simulation()
    
    
    plot_results(democrat_scores, republican_scores, neutral_scores)
    
    print("\nSimulation complete! Results plotted to 'gun_violence_attitude_scores.png'")

if __name__ == "__main__":
    main()
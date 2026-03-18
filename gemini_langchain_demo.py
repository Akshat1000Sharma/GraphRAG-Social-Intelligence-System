"""
gemini_langchain_demo.py
========================
Standalone demo of the LangChain + Gemini + PydanticOutputParser integration.
Mirrors the sample code pattern exactly, extended for all four insight schemas
used in the Social Graph Intelligence system.

Run:
    python gemini_langchain_demo.py

Requires:
    pip install langchain langchain-google-genai google-generativeai pydantic python-dotenv
    GOOGLE_API_KEY=your_key in .env
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List

load_dotenv()

# ─── Model ────────────────────────────────────────────────────────────────────
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")


# ─── Schema 1: General Graph Insight ─────────────────────────────────────────
class GraphInsight(BaseModel):
    summary: str = Field(description="A concise 2-3 sentence summary directly answering the user's query")
    key_findings: List[str] = Field(description="List of 2-4 specific findings grounded in the graph data")
    confidence_assessment: str = Field(description="Brief assessment of result confidence based on data quality")
    recommended_action: str = Field(description="One actionable recommendation for the user")


# ─── Schema 2: Influencer Detection ──────────────────────────────────────────
class InfluencerInsight(BaseModel):
    summary: str = Field(description="Summary of the user's influence in the network")
    influence_factors: List[str] = Field(description="List of factors driving their influence score")
    network_role: str = Field(description="The user's role: regular_user, influencer, content_creator, or community_hub")
    comparison: str = Field(description="How this user compares to the average network participant")


# ─── Schema 3: Connection Explanation ────────────────────────────────────────
class ConnectionExplanation(BaseModel):
    relationship_summary: str = Field(description="Natural language summary of how the two users are connected")
    connection_strength: str = Field(description="Assessment of connection strength: strong, moderate, or weak")
    common_ground: List[str] = Field(description="List of shared interests, friends, or communities")
    recommendation: str = Field(description="Whether connecting is recommended and why")


# ─── Schema 4: Trending Posts ─────────────────────────────────────────────────
class TrendingInsight(BaseModel):
    summary: str = Field(description="Overview of current trending topics")
    top_themes: List[str] = Field(description="List of dominant themes across trending posts")
    engagement_pattern: str = Field(description="Description of the engagement pattern observed")
    peak_topic: str = Field(description="The single most engaging topic right now")


# ─── Demo 1: Friend Recommendation Insight (PydanticOutputParser) ─────────────
print("\n" + "="*60)
print("DEMO 1: Friend Recommendation → GraphInsight (Pydantic)")
print("="*60)

parser1 = PydanticOutputParser(pydantic_object=GraphInsight)

template1 = PromptTemplate(
    template=(
        "You are a social network analyst.\n"
        "User query: {user_query}\n"
        "Graph data: {graph_data}\n\n"
        "{format_instructions}"
    ),
    input_variables=["user_query", "graph_data"],
    partial_variables={"format_instructions": parser1.get_format_instructions()},
)

chain1 = template1 | model | parser1

result1: GraphInsight = chain1.invoke({
    "user_query": "Recommend new friends for user_1",
    "graph_data": (
        "user_1 has 5 friends. "
        "user_7 shares 3 mutual friends with user_1. "
        "user_12 shares 2 mutual friends. "
        "GNN link probability: user_7=0.91, user_12=0.84."
    ),
})

print(f"Summary:             {result1.summary}")
print(f"Key Findings:        {result1.key_findings}")
print(f"Confidence:          {result1.confidence_assessment}")
print(f"Recommended Action:  {result1.recommended_action}")
print(f"Type:                {type(result1)}")


# ─── Demo 2: Influencer Detection (PydanticOutputParser) ──────────────────────
print("\n" + "="*60)
print("DEMO 2: Influencer Detection → InfluencerInsight (Pydantic)")
print("="*60)

parser2 = PydanticOutputParser(pydantic_object=InfluencerInsight)

template2 = PromptTemplate(
    template=(
        "You are a social network analyst specializing in influence measurement.\n"
        "User query: {user_query}\n"
        "User stats: {user_stats}\n\n"
        "{format_instructions}"
    ),
    input_variables=["user_query", "user_stats"],
    partial_variables={"format_instructions": parser2.get_format_instructions()},
)

chain2 = template2 | model | parser2

result2: InfluencerInsight = chain2.invoke({
    "user_query": "What is the influence of user_3 in the network?",
    "user_stats": (
        "user_3: 1200 followers, 45 posts, avg 320 likes/post, "
        "GNN influence_score=0.87, predicted_class=influencer (confidence=0.92)."
    ),
})

print(f"Summary:           {result2.summary}")
print(f"Influence Factors: {result2.influence_factors}")
print(f"Network Role:      {result2.network_role}")
print(f"Comparison:        {result2.comparison}")
print(f"Type:              {type(result2)}")


# ─── Demo 3: Connection Explanation (PydanticOutputParser) ────────────────────
print("\n" + "="*60)
print("DEMO 3: Explain Connection → ConnectionExplanation (Pydantic)")
print("="*60)

parser3 = PydanticOutputParser(pydantic_object=ConnectionExplanation)

template3 = PromptTemplate(
    template=(
        "You are a social network analyst explaining user relationships.\n"
        "User query: {user_query}\n"
        "Connection data: {connection_data}\n\n"
        "{format_instructions}"
    ),
    input_variables=["user_query", "connection_data"],
    partial_variables={"format_instructions": parser3.get_format_instructions()},
)

chain3 = template3 | model | parser3

result3: ConnectionExplanation = chain3.invoke({
    "user_query": "Explain the connection between user_1 and user_9",
    "connection_data": (
        "Shortest path: user_1 -> user_4 -> user_9 (2 hops). "
        "Common friends: user_4, user_6. "
        "Both liked posts about AI and Sports. "
        "Both members of group 'Tech Enthusiasts'."
    ),
})

print(f"Relationship:      {result3.relationship_summary}")
print(f"Strength:          {result3.connection_strength}")
print(f"Common Ground:     {result3.common_ground}")
print(f"Recommendation:    {result3.recommendation}")
print(f"Type:              {type(result3)}")


# ─── Demo 4: Trending Posts (PydanticOutputParser) ────────────────────────────
print("\n" + "="*60)
print("DEMO 4: Trending Posts → TrendingInsight (Pydantic)")
print("="*60)

parser4 = PydanticOutputParser(pydantic_object=TrendingInsight)

template4 = PromptTemplate(
    template=(
        "You are a social media trend analyst.\n"
        "User query: {user_query}\n"
        "Trending data: {trending_data}\n\n"
        "{format_instructions}"
    ),
    input_variables=["user_query", "trending_data"],
    partial_variables={"format_instructions": parser4.get_format_instructions()},
)

chain4 = template4 | model | parser4

result4: TrendingInsight = chain4.invoke({
    "user_query": "What are the trending posts right now?",
    "trending_data": (
        "Post 1 (AI, 4200 likes, 2h old), "
        "Post 2 (Sports, 3800 likes, 3h old), "
        "Post 3 (AI, 3100 likes, 1h old), "
        "Post 4 (Music, 2900 likes, 4h old), "
        "Post 5 (Tech, 2700 likes, 2h old)."
    ),
})

print(f"Summary:            {result4.summary}")
print(f"Top Themes:         {result4.top_themes}")
print(f"Engagement Pattern: {result4.engagement_pattern}")
print(f"Peak Topic:         {result4.peak_topic}")
print(f"Type:               {type(result4)}")


# ─── Demo 5: Simple chain (StrOutputParser) ───────────────────────────────────
print("\n" + "="*60)
print("DEMO 5: Simple fallback chain → StrOutputParser")
print("="*60)

str_parser = StrOutputParser()

template5 = PromptTemplate(
    template="Summarize this social graph insight in one sentence: {insight}",
    input_variables=["insight"],
)

chain5 = template5 | model | str_parser

result5: str = chain5.invoke({
    "insight": "User_1 has 5 mutual friends with user_7 and a GNN link probability of 0.91.",
})

print(f"Summary: {result5}")
print(f"Type:    {type(result5)}")


# ─── Demo 6: Two-step chaining (report → summary) ─────────────────────────────
print("\n" + "="*60)
print("DEMO 6: Two-step chain (graph analysis → executive summary)")
print("="*60)

template_analysis = PromptTemplate(
    template="Analyze this social graph data and write a detailed report: {data}",
    input_variables=["data"],
)

template_summary = PromptTemplate(
    template="Summarize this social graph report in exactly 2 sentences: {report}",
    input_variables=["report"],
)

# Two-step: analysis → summary (using StrOutputParser between steps)
analysis_chain = template_analysis | model | str_parser
summary_chain = template_summary | model | str_parser

graph_data = (
    "Network has 500 users. Top influencer: user_3 (score=0.87). "
    "Most liked post: 'AI trends 2025' (4200 likes). "
    "Average friend count: 12. Community hubs: user_3, user_7, user_15."
)

analysis = analysis_chain.invoke({"data": graph_data})
summary = summary_chain.invoke({"report": analysis})

print(f"Executive Summary:\n{summary}")

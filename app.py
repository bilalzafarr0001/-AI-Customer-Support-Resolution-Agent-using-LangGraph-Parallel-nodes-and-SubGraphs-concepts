import os
from uuid import uuid4
from typing import TypedDict, Literal

import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, interrupt


# ============================================================
# 1. ENVIRONMENT
# ============================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv(
    "MODEL_NAME",
    "openai/gpt-oss-20b"
)


# ============================================================
# 2. LLM
# ============================================================

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY is missing from your .env file.")
    st.stop()


llm = ChatGroq(
    model=MODEL_NAME,
    temperature=0
)


# ============================================================
# 3. STRUCTURED OUTPUT MODELS
# ============================================================
#
# This is very important.
#
# Instead of asking the LLM:
#
# "Tell me what you think about this ticket."
#
# We force the LLM to return a predictable structure.
#
# This makes the result much easier for LangGraph to use.
# ============================================================


class TicketTriage(BaseModel):
    category: Literal[
        "technical",
        "billing",
        "account",
        "general"
    ] = Field(
        description="The main category of the customer ticket."
    )

    priority: Literal[
        "low",
        "medium",
        "high",
        "critical"
    ] = Field(
        description="Urgency of the customer issue."
    )

    needs_human: bool = Field(
        description=(
            "True when human approval is required, "
            "especially for sensitive, risky, financial, "
            "security, or uncertain cases."
        )
    )

    summary: str = Field(
        description="Short summary of the customer issue."
    )

    recommended_action: str = Field(
        description="Recommended action for support."
    )

    confidence: float = Field(
        description="Confidence from 0 to 1."
    )


class TechnicalReview(BaseModel):
    root_cause: str = Field(
        description="Likely technical cause of the issue."
    )

    troubleshooting_steps: list[str] = Field(
        description="Recommended troubleshooting steps."
    )

    technical_risk: Literal[
        "low",
        "medium",
        "high"
    ] = Field(
        description=(
            "MUST be exactly one of these values: "
            "low, medium, high. "
            "Do not add any explanation to this field."
        )
    )

    recommendation: str = Field(
        description="Recommended technical action."
    )


class SecurityReview(BaseModel):
    security_risk: Literal[
        "low",
        "medium",
        "high",
        "critical"
    ]

    security_concerns: list[str]

    policy_concerns: list[str]

    recommendation: str


class CustomerExperienceReview(BaseModel):
    customer_sentiment: Literal[
        "positive",
        "neutral",
        "frustrated",
        "angry"
    ]

    tone: Literal[
        "friendly",
        "empathetic",
        "formal",
        "apologetic"
    ]

    customer_needs: list[str]

    recommendation: str


class FinalResponse(BaseModel):
    subject: str = Field(
        description="Short subject line for the customer response."
    )

    response: str = Field(
        description="Complete response that should be sent to the customer."
    )

    next_steps: list[str] = Field(
        description="Clear next steps for the customer."
    )

    escalation_required: bool = Field(
        description="Whether this case still requires escalation."
    )


# ============================================================
# 4. PROMPTS
# ============================================================
#
# Keep prompts separate from graph logic.
#
# This is much better than putting giant prompt strings
# directly inside node functions.
# ============================================================


TRIAGE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an expert customer support triage agent.

Analyze the customer ticket carefully.

Your job is to determine:

1. Category
2. Priority
3. Whether human approval is required
4. Short summary
5. Recommended action
6. Confidence

Human approval should normally be required when:
- There is a security concern
- There is a financial/refund issue
- Sensitive account information is involved
- The customer may suffer significant loss
- The situation is ambiguous
- The recommended action could have important consequences

Do not invent facts.
Only use information present in the ticket.
"""
        ),
        (
            "human",
            """
Customer ticket:

{ticket}
"""
        )
    ]
)


TECHNICAL_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a senior software support engineer.

Analyze the customer issue from a technical perspective.

Identify:
- likely root cause
- troubleshooting steps
- technical risk
- recommended technical action

IMPORTANT:

The technical_risk field MUST contain ONLY ONE of:

low
medium
high

Do NOT put an explanation inside technical_risk.

CORRECT:
"technical_risk": "medium"

WRONG:
"technical_risk": "medium - the issue may cause..."

If you need to explain why the risk is medium,
put that explanation in the recommendation field.

Do not invent system behavior.
Keep recommendations practical and safe.
"""
        ),
        (
            "human",
            """
Customer ticket:

{ticket}

Triage information:

{triage}
"""
        )
    ]
)


SECURITY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a security and support-policy expert.

Analyze the customer ticket for:

- security risks
- account risks
- privacy concerns
- financial/policy concerns
- situations requiring human approval

Do not assume facts that are not present.

If there is no obvious risk, clearly say so.
"""
        ),
        (
            "human",
            """
Customer ticket:

{ticket}

Triage information:

{triage}
"""
        )
    ]
)


CUSTOMER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an expert customer experience specialist.

Analyze the ticket from the customer's perspective.

Determine:

- customer sentiment
- appropriate communication tone
- customer needs
- recommended communication strategy

The final communication should be professional,
clear, empathetic, and not overly verbose.
"""
        ),
        (
            "human",
            """
Customer ticket:

{ticket}

Triage information:

{triage}
"""
        )
    ]
)


FINAL_RESPONSE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a senior customer support response writer.

Create the final response to the customer.

You MUST provide ALL FOUR fields:

1. subject
2. response
3. next_steps
4. escalation_required

The subject must always be present.

Use the ticket, triage, technical review,
security review, customer experience review,
and human decision.

Rules:

- Never invent facts.
- Do not promise something the company cannot guarantee.
- Be professional and empathetic.
- Clearly explain the next steps.
- If human approval was rejected,
  follow the human feedback.
- If human approval was accepted,
  follow the approved direction.
- If escalation is required, make that clear.
"""
        ),
        (
            "human",
            """
CUSTOMER TICKET:
{ticket}

TRIAGE:
{triage}

TECHNICAL REVIEW:
{technical}

SECURITY REVIEW:
{security}

CUSTOMER EXPERIENCE REVIEW:
{customer}

HUMAN DECISION:
{human_decision}
"""
        )
    ]
)

# ============================================================
# 5. STRUCTURED LLM CHAINS
# ============================================================

triage_chain = (
    TRIAGE_PROMPT
    | llm.with_structured_output(TicketTriage)
)

technical_chain = (
    TECHNICAL_PROMPT
    | llm.with_structured_output(TechnicalReview)
)

security_chain = (
    SECURITY_PROMPT
    | llm.with_structured_output(SecurityReview)
)

customer_chain = (
    CUSTOMER_PROMPT
    | llm.with_structured_output(CustomerExperienceReview)
)

final_response_chain = (
    FINAL_RESPONSE_PROMPT
    | llm.with_structured_output(
        FinalResponse,
        method="json_schema",
        strict=True
    )
)


# ============================================================
# 6. SUBGRAPH STATES
# ============================================================


class TechnicalState(TypedDict):
    ticket: str
    triage: TicketTriage
    review: TechnicalReview | None


class SecurityState(TypedDict):
    ticket: str
    triage: TicketTriage
    review: SecurityReview | None


class CustomerState(TypedDict):
    ticket: str
    triage: TicketTriage
    review: CustomerExperienceReview | None


# ============================================================
# 7. MAIN GRAPH STATE
# ============================================================


class SupportState(TypedDict):
    ticket: str

    triage: TicketTriage | None

    technical_review: TechnicalReview | None

    security_review: SecurityReview | None

    customer_review: CustomerExperienceReview | None

    human_decision: dict | None

    final_response: FinalResponse | None


# ============================================================
# 8. TECHNICAL SUBGRAPH
# ============================================================


def technical_analysis(
    state: TechnicalState
):
    result = technical_chain.invoke(
        {
            "ticket": state["ticket"],
            "triage": state["triage"].model_dump_json()
        }
    )

    return {
        "review": result
    }


technical_builder = StateGraph(
    TechnicalState
)

technical_builder.add_node(
    "technical_analysis",
    technical_analysis
)

technical_builder.add_edge(
    START,
    "technical_analysis"
)

technical_builder.add_edge(
    "technical_analysis",
    END
)

technical_subgraph = technical_builder.compile()


# ============================================================
# 9. SECURITY SUBGRAPH
# ============================================================


def security_analysis(
    state: SecurityState
):
    result = security_chain.invoke(
        {
            "ticket": state["ticket"],
            "triage": state["triage"].model_dump_json()
        }
    )

    return {
        "review": result
    }


security_builder = StateGraph(
    SecurityState
)

security_builder.add_node(
    "security_analysis",
    security_analysis
)

security_builder.add_edge(
    START,
    "security_analysis"
)

security_builder.add_edge(
    "security_analysis",
    END
)

security_subgraph = security_builder.compile()


# ============================================================
# 10. CUSTOMER EXPERIENCE SUBGRAPH
# ============================================================


def customer_analysis(
    state: CustomerState
):
    result = customer_chain.invoke(
        {
            "ticket": state["ticket"],
            "triage": state["triage"].model_dump_json()
        }
    )

    return {
        "review": result
    }


customer_builder = StateGraph(
    CustomerState
)

customer_builder.add_node(
    "customer_analysis",
    customer_analysis
)

customer_builder.add_edge(
    START,
    "customer_analysis"
)

customer_builder.add_edge(
    "customer_analysis",
    END
)



customer_subgraph = customer_builder.compile()


# ============================================================
# 11. MAIN GRAPH - TRIAGE NODE
# ============================================================


def triage_ticket(
    state: SupportState
):
    result = triage_chain.invoke(
        {
            "ticket": state["ticket"]
        }
    )

    return {
        "triage": result
    }


# ============================================================
# 12. MAIN GRAPH - PARALLEL TECHNICAL NODE
# ============================================================


def run_technical_review(
    state: SupportState
):
    result = technical_subgraph.invoke(
        {
            "ticket": state["ticket"],
            "triage": state["triage"],
            "review": None
        }
    )

    return {
        "technical_review": result["review"]
    }


# ============================================================
# 13. MAIN GRAPH - PARALLEL SECURITY NODE
# ============================================================


def run_security_review(
    state: SupportState
):
    result = security_subgraph.invoke(
        {
            "ticket": state["ticket"],
            "triage": state["triage"],
            "review": None
        }
    )

    return {
        "security_review": result["review"]
    }


# ============================================================
# 14. MAIN GRAPH - PARALLEL CUSTOMER NODE
# ============================================================


def run_customer_review(
    state: SupportState
):
    result = customer_subgraph.invoke(
        {
            "ticket": state["ticket"],
            "triage": state["triage"],
            "review": None
        }
    )

    return {
        "customer_review": result["review"]
    }


# ============================================================
# 15. CONDITIONAL ROUTING
# ============================================================
#
# IMPORTANT:
#
# We are NOT using Command(goto=...)
# for routing.
#
# We use add_conditional_edges().
# ============================================================


def route_after_reviews(
    state: SupportState
):
    triage = state["triage"]

    if triage is None:
        return "human_review"

    if triage.needs_human:
        return "human_review"

    if triage.priority in ["high", "critical"]:
        return "human_review"

    if state["security_review"]:
        if state["security_review"].security_risk in [
            "high",
            "critical"
        ]:
            return "human_review"

    return "automatic_response"


# ============================================================
# 16. HUMAN-IN-THE-LOOP
# ============================================================
#
# interrupt() pauses the graph.
#
# The graph will wait here until Streamlit resumes it.
# ============================================================


def human_review(
    state: SupportState
):
    triage = state["triage"]

    review_payload = {
        "message": (
            "This support ticket requires human approval "
            "before a final response is generated."
        ),

        "ticket": state["ticket"],

        "priority": (
            triage.priority
            if triage
            else "unknown"
        ),

        "reason": (
            triage.recommended_action
            if triage
            else "Review required."
        ),

        "technical_review": (
            state["technical_review"].model_dump()
            if state["technical_review"]
            else {}
        ),

        "security_review": (
            state["security_review"].model_dump()
            if state["security_review"]
            else {}
        )
    }

    decision = interrupt(
        review_payload
    )

    return {
        "human_decision": decision
    }


# ============================================================
# 17. FINAL RESPONSE
# ============================================================


def generate_final_response(
    state: SupportState
):
    result = final_response_chain.invoke(
        {
            "ticket": state["ticket"],

            "triage": (
                state["triage"].model_dump_json()
                if state["triage"]
                else "{}"
            ),

            "technical": (
                state["technical_review"].model_dump_json()
                if state["technical_review"]
                else "{}"
            ),

            "security": (
                state["security_review"].model_dump_json()
                if state["security_review"]
                else "{}"
            ),

            "customer": (
                state["customer_review"].model_dump_json()
                if state["customer_review"]
                else "{}"
            ),

            "human_decision": str(
                state["human_decision"]
            )
        }
    )

    return {
        "final_response": result
    }


# ============================================================
# 18. BUILD MAIN GRAPH
# ============================================================


builder = StateGraph(
    SupportState
)


# ------------------------------------------------------------
# Nodes
# ------------------------------------------------------------

builder.add_node(
    "triage_ticket",
    triage_ticket
)

builder.add_node(
    "technical_review",
    run_technical_review
)

builder.add_node(
    "security_review",
    run_security_review
)

builder.add_node(
    "customer_review",
    run_customer_review
)

builder.add_node(
    "human_review",
    human_review
)

builder.add_node(
    "generate_final_response",
    generate_final_response
)


# ============================================================
# 19. GRAPH EDGES
# ============================================================


# START
#   ↓
# TRIAGE

builder.add_edge(
    START,
    "triage_ticket"
)


# ============================================================
# PARALLEL FAN-OUT
# ============================================================
#
# After triage, all three nodes receive the same state.
#
# They can execute independently.
#
#                    triage
#                   /  |  \
#                  /   |   \
#          technical security customer
#

builder.add_edge(
    "triage_ticket",
    "technical_review"
)

builder.add_edge(
    "triage_ticket",
    "security_review"
)

builder.add_edge(
    "triage_ticket",
    "customer_review"
)


# ============================================================
# FAN-IN + CONDITIONAL ROUTING
# ============================================================
#
# All three branches converge here logically.
#
# LangGraph waits until all upstream branches complete
# before executing this conditional routing point.
# ============================================================


builder.add_conditional_edges(
    "technical_review",
    lambda state: "continue",
    {
        "continue": "generate_final_response"
    }
)

builder.add_conditional_edges(
    "security_review",
    lambda state: "continue",
    {
        "continue": "generate_final_response"
    }
)

builder.add_conditional_edges(
    "customer_review",
    lambda state: "continue",
    {
        "continue": "generate_final_response"
    }
)


# ============================================================
# IMPORTANT
# ============================================================
#
# The above direct convergence is not where we want our
# human conditional decision.
#
# So instead, we use a dedicated review-completion node.
#
# To keep the example simple, we will rebuild this part below.
# ============================================================


def review_complete(
    state: SupportState
):
    return {}


# Rebuild the graph cleanly.

builder = StateGraph(
    SupportState
)


builder.add_node(
    "triage_ticket",
    triage_ticket
)

builder.add_node(
    "technical_review",
    run_technical_review
)

builder.add_node(
    "security_review",
    run_security_review
)

builder.add_node(
    "customer_review",
    run_customer_review
)

builder.add_node(
    "review_complete",
    review_complete
)

builder.add_node(
    "human_review",
    human_review
)

builder.add_node(
    "generate_final_response",
    generate_final_response
)


# START → TRIAGE

builder.add_edge(
    START,
    "triage_ticket"
)


# TRIAGE → PARALLEL NODES

builder.add_edge(
    "triage_ticket",
    "technical_review"
)

builder.add_edge(
    "triage_ticket",
    "security_review"
)

builder.add_edge(
    "triage_ticket",
    "customer_review"
)


# PARALLEL NODES → FAN-IN

builder.add_edge(
    "technical_review",
    "review_complete"
)

builder.add_edge(
    "security_review",
    "review_complete"
)

builder.add_edge(
    "customer_review",
    "review_complete"
)


# ============================================================
# CONDITIONAL EDGE
# ============================================================

builder.add_conditional_edges(
    "review_complete",
    route_after_reviews,
    {
        "human_review": "human_review",
        "automatic_response": "generate_final_response"
    }
)


# HUMAN → FINAL RESPONSE

builder.add_edge(
    "human_review",
    "generate_final_response"
)


# FINAL → END

builder.add_edge(
    "generate_final_response",
    END
)


# ============================================================
# CHECKPOINTER
# ============================================================
#
# Required for interrupt()/resume().
# ============================================================

@st.cache_resource
def get_graph():

    checkpointer = InMemorySaver()

    return builder.compile(
        checkpointer=checkpointer
    )


graph = get_graph()
 

# ============================================================
# 20. STREAMLIT UI
# ============================================================


st.set_page_config(
    page_title="AI Customer Support Agent",
    layout="wide"
)

st.title(
    "🤖 AI Customer Support Resolution Agent"
)

st.caption(
    "LangGraph: Structured Output + Parallel Nodes + "
    "SubGraphs + Conditional Edges + Human-in-the-Loop"
)


# ============================================================
# SESSION STATE
# ============================================================


if "thread_id" not in st.session_state:
    st.session_state.thread_id = None

if "graph_result" not in st.session_state:
    st.session_state.graph_result = None

if "running_ticket" not in st.session_state:
    st.session_state.running_ticket = None


# ============================================================
# NEW TICKET
# ============================================================


ticket = st.text_area(
    "Customer Support Ticket",
    height=180,
    placeholder=(
        "Example:\n"
        "I was charged twice for my subscription. "
        "I can see two payments on my bank statement "
        "and I want one of them refunded."
    )
)


if st.button(
    "🚀 Process Ticket"
):

    if not ticket.strip():
        st.warning(
            "Please enter a customer ticket."
        )
        st.stop()

    # Create a new conversation/thread.

    thread_id = str(uuid4())

    st.session_state.thread_id = thread_id
    st.session_state.running_ticket = ticket.strip()

    initial_state = {
        "ticket": ticket.strip(),

        "triage": None,

        "technical_review": None,

        "security_review": None,

        "customer_review": None,

        "human_decision": None,

        "final_response": None
    }

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    with st.spinner(
        "Running LangGraph..."
    ):

        result = graph.invoke(
            initial_state,
            config
        )

    st.session_state.graph_result = result

    st.rerun()


# ============================================================
# DISPLAY CURRENT RESULT
# ============================================================


result = st.session_state.graph_result


if result:

    # --------------------------------------------------------
    # TRIAGE
    # --------------------------------------------------------

    if result.get("triage"):

        st.divider()

        st.subheader(
            "🧠 LLM Triage"
        )

        triage = result["triage"]

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Category",
                triage.category
            )

        with col2:
            st.metric(
                "Priority",
                triage.priority
            )

        with col3:
            st.metric(
                "Human Required",
                str(triage.needs_human)
            )

        with col4:
            st.metric(
                "Confidence",
                f"{triage.confidence:.2f}"
            )

        st.write(
            "**Summary:**",
            triage.summary
        )

        st.write(
            "**Recommended Action:**",
            triage.recommended_action
        )


    # --------------------------------------------------------
    # PARALLEL REVIEWS
    # --------------------------------------------------------

    if (
        result.get("technical_review")
        and result.get("security_review")
        and result.get("customer_review")
    ):

        st.divider()

        st.subheader(
            "⚡ Parallel Expert Reviews"
        )

        col1, col2, col3 = st.columns(3)


        # Technical

        with col1:

            st.markdown(
                "### 🔧 Technical"
            )

            technical = (
                result["technical_review"]
            )

            st.write(
                "**Root Cause**"
            )

            st.write(
                technical.root_cause
            )

            st.write(
                "**Risk:**",
                technical.technical_risk
            )

            st.write(
                "**Troubleshooting:**"
            )

            for step in technical.troubleshooting_steps:
                st.write(
                    f"- {step}"
                )


        # Security

        with col2:

            st.markdown(
                "### 🔐 Security"
            )

            security = (
                result["security_review"]
            )

            st.write(
                "**Risk:**",
                security.security_risk
            )

            st.write(
                "**Security Concerns:**"
            )

            for item in security.security_concerns:
                st.write(
                    f"- {item}"
                )

            st.write(
                "**Policy Concerns:**"
            )

            for item in security.policy_concerns:
                st.write(
                    f"- {item}"
                )


        # Customer

        with col3:

            st.markdown(
                "### 😊 Customer"
            )

            customer = (
                result["customer_review"]
            )

            st.write(
                "**Sentiment:**",
                customer.customer_sentiment
            )

            st.write(
                "**Tone:**",
                customer.tone
            )

            st.write(
                "**Customer Needs:**"
            )

            for item in customer.customer_needs:
                st.write(
                    f"- {item}"
                )


    # ========================================================
    # HUMAN APPROVAL
    # ========================================================

    interrupts = result.get(
        "__interrupt__"
    )


    if interrupts:

        st.divider()

        st.warning(
            "⚠️ Human approval is required."
        )

        interrupt_value = interrupts[0].value

        st.write(
            interrupt_value["message"]
        )

        st.write(
            "**Reason:**",
            interrupt_value["reason"]
        )

        st.write(
            "**Priority:**",
            interrupt_value["priority"]
        )

        feedback = st.text_area(
            "Optional human feedback",
            placeholder=(
                "Example: Approve the refund, "
                "but ask the customer to confirm "
                "the last four digits of the card."
            )
        )


        col1, col2 = st.columns(2)


        with col1:

            if st.button(
                "✅ Approve"
            ):

                config = {
                    "configurable": {
                        "thread_id": (
                            st.session_state.thread_id
                        )
                    }
                }

                with st.spinner(
                    "Resuming LangGraph..."
                ):

                    resumed_result = graph.invoke(
                        Command(
                            resume={
                                "approved": True,
                                "feedback": feedback
                            }
                        ),
                        config
                    )

                st.session_state.graph_result = (
                    resumed_result
                )

                st.rerun()


        with col2:

            if st.button(
                "❌ Reject"
            ):

                config = {
                    "configurable": {
                        "thread_id": (
                            st.session_state.thread_id
                        )
                    }
                }

                with st.spinner(
                    "Resuming LangGraph..."
                ):

                    resumed_result = graph.invoke(
                        Command(
                            resume={
                                "approved": False,
                                "feedback": feedback
                            }
                        ),
                        config
                    )

                st.session_state.graph_result = (
                    resumed_result
                )

                st.rerun()


    # ========================================================
    # FINAL RESPONSE
    # ========================================================

    if result.get(
        "final_response"
    ):

        st.divider()

        st.subheader(
            "📨 Final Customer Response"
        )

        final_response = (
            result["final_response"]
        )

        st.markdown(
            f"### {final_response.subject}"
        )

        st.write(
            final_response.response
        )

        st.markdown(
            "### Next Steps"
        )

        for step in final_response.next_steps:

            st.write(
                f"- {step}"
            )

        if final_response.escalation_required:

            st.warning(
                "This case requires escalation."
            )

        else:

            st.success(
                "No further escalation required."
            )
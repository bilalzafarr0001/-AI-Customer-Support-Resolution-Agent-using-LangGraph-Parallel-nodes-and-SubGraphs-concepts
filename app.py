import streamlit as st
from typing import TypedDict

from langgraph.graph import StateGraph, START, END


# ============================================================
# 1. SUBGRAPH STATES
# ============================================================

class TechnicalState(TypedDict):
    project: str
    analysis: str


class SecurityState(TypedDict):
    project: str
    analysis: str


class UXState(TypedDict):
    project: str
    analysis: str


# ============================================================
# 2. MAIN GRAPH STATE
# ============================================================

class MainState(TypedDict):
    project: str
    technical_review: str
    security_review: str
    ux_review: str
    final_report: str


# ============================================================
# 3. TECHNICAL SUBGRAPH
# ============================================================

def technical_analyze(state: TechnicalState):
    project = state["project"]

    return {
        "analysis": (
            f"Technical Review:\n"
            f"- Project: {project}\n"
            f"- Backend API will be required.\n"
            f"- Database will be required.\n"
            f"- Authentication should be implemented.\n"
            f"- Application should have a scalable architecture.\n"
            f"- API and frontend should be separated."
        )
    }


technical_builder = StateGraph(TechnicalState)

technical_builder.add_node(
    "technical_analyze",
    technical_analyze
)

technical_builder.add_edge(
    START,
    "technical_analyze"
)

technical_builder.add_edge(
    "technical_analyze",
    END
)

technical_subgraph = technical_builder.compile()


# ============================================================
# 4. SECURITY SUBGRAPH
# ============================================================

def security_analyze(state: SecurityState):
    project = state["project"]

    return {
        "analysis": (
            f"Security Review:\n"
            f"- Project: {project}\n"
            f"- Authentication is required.\n"
            f"- Passwords must be securely hashed.\n"
            f"- API endpoints need authorization checks.\n"
            f"- User input must be validated.\n"
            f"- Sensitive data should be encrypted."
        )
    }


security_builder = StateGraph(SecurityState)

security_builder.add_node(
    "security_analyze",
    security_analyze
)

security_builder.add_edge(
    START,
    "security_analyze"
)

security_builder.add_edge(
    "security_analyze",
    END
)

security_subgraph = security_builder.compile()


# ============================================================
# 5. UX SUBGRAPH
# ============================================================

def ux_analyze(state: UXState):
    project = state["project"]

    return {
        "analysis": (
            f"UX Review:\n"
            f"- Project: {project}\n"
            f"- Navigation should be simple.\n"
            f"- Important actions should be easy to find.\n"
            f"- Loading states should be visible.\n"
            f"- Forms should provide useful validation.\n"
            f"- UI should work well on mobile devices."
        )
    }


ux_builder = StateGraph(UXState)

ux_builder.add_node(
    "ux_analyze",
    ux_analyze
)

ux_builder.add_edge(
    START,
    "ux_analyze"
)

ux_builder.add_edge(
    "ux_analyze",
    END
)

ux_subgraph = ux_builder.compile()


# ============================================================
# 6. MAIN GRAPH NODES
# ============================================================

def technical_review(state: MainState):
    result = technical_subgraph.invoke({
        "project": state["project"],
        "analysis": ""
    })

    return {
        "technical_review": result["analysis"]
    }


def security_review(state: MainState):
    result = security_subgraph.invoke({
        "project": state["project"],
        "analysis": ""
    })

    return {
        "security_review": result["analysis"]
    }


def ux_review(state: MainState):
    result = ux_subgraph.invoke({
        "project": state["project"],
        "analysis": ""
    })

    return {
        "ux_review": result["analysis"]
    }


# ============================================================
# 7. FAN-IN NODE
# ============================================================

def combine_results(state: MainState):

    final_report = f"""
# AI Project Review

## Project

{state["project"]}

---

## Technical Review

{state["technical_review"]}

---

## Security Review

{state["security_review"]}

---

## UX Review

{state["ux_review"]}

---

## Final Conclusion

The project should be designed by considering technical,
security, and user-experience requirements together.
"""

    return {
        "final_report": final_report
    }


# ============================================================
# 8. BUILD MAIN GRAPH
# ============================================================

builder = StateGraph(MainState)


# Main graph nodes
builder.add_node(
    "technical_review",
    technical_review
)

builder.add_node(
    "security_review",
    security_review
)

builder.add_node(
    "ux_review",
    ux_review
)

builder.add_node(
    "combine_results",
    combine_results
)


# ============================================================
# 9. PARALLEL EXECUTION
# ============================================================

builder.add_edge(
    START,
    "technical_review"
)

builder.add_edge(
    START,
    "security_review"
)

builder.add_edge(
    START,
    "ux_review"
)


# ============================================================
# 10. FAN-IN
# ============================================================

builder.add_edge(
    "technical_review",
    "combine_results"
)

builder.add_edge(
    "security_review",
    "combine_results"
)

builder.add_edge(
    "ux_review",
    "combine_results"
)


builder.add_edge(
    "combine_results",
    END
)


graph = builder.compile()


# ============================================================
# 11. STREAMLIT UI
# ============================================================

st.set_page_config(
    page_title="Parallel SubGraph Agent",
    layout="wide"
)

st.title("🚀 Parallel SubGraph Project Reviewer")

st.write(
    "This project demonstrates Parallel Nodes, "
    "SubGraphs, Fan-out and Fan-in."
)

project = st.text_area(
    "Describe your software project",
    placeholder=(
        "Example: Build an e-commerce application "
        "with AI product recommendations."
    ),
    height=150
)


if st.button("Run Project Review") and project.strip():

    initial_state = {
        "project": project.strip(),
        "technical_review": "",
        "security_review": "",
        "ux_review": "",
        "final_report": ""
    }

    with st.spinner("Running parallel expert reviews..."):

        result = graph.invoke(initial_state)

    st.success("Review completed!")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("🔧 Technical")
        st.write(result["technical_review"])

    with col2:
        st.subheader("🔐 Security")
        st.write(result["security_review"])

    with col3:
        st.subheader("🎨 UX")
        st.write(result["ux_review"])

    st.divider()

    st.subheader("📋 Final Report")

    st.markdown(result["final_report"])
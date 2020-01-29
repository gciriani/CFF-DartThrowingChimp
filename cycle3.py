import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
sns.set(rc={'figure.figsize':(11, 4)})

cycle3 = pd.read_csv('Cycles3_3.csv')

def preprocess(cycle3):
    cycle3 = cycle3.drop(columns={'Unnamed: 0'})
    cycle3 = cycle3.groupby('Timestamp')['Timestamp', 'Question', 'TeamName',
                                         'SWRPS', 'FairSkill', 'Forecaster'].first()

    cycle3.loc[:, 'Timestamp'] = pd.to_datetime(cycle3.Timestamp.str[:-38])
    cycle3 = cycle3.set_index("Timestamp")

    cycle3 = cycle3.sort_index()
    return cycle3


def plot_ts(data, title):
    ax = sns.lineplot(
        x=data.index,
        y=data.SWRPS,
        data=data,
        dashes=False,
        marker='o',
        label=title
    )
    ax.set(ylim=(0,1.3))
    plt.title(question)

processed = preprocess(cycle3)
# processed
# teams = processed.pivot_table(index='Timestamp', values='SWRPS', columns='TeamName')
# teams

st.title("Resampled Mean Score By CFF")
question = st.sidebar.selectbox(
    "Counterfactual Forecast",
    sorted(processed.Question.unique()))
# for question in processed.Question.unique():
resampling_period = st.sidebar.radio(
    "Resampling Period",
    ['H (hourly)', 'D (daily)', 'B (business daily)'])
resampling_period = resampling_period.split(' ')[0]
scoring_method = st.sidebar.radio(
    "Scoring Method",
    ['FairSkill', 'SWRPS'])

fig = go.Figure()
kiwi = processed.query(
    "Question == @question & TeamName == 'Kiwi'"
).resample(resampling_period).mean()
kiwi_marker_size = processed.query(
    "Question == @question & TeamName == 'Kiwi'"
).resample(resampling_period).count().Forecaster

mango = processed.query(
    "Question == @question & TeamName == 'Mango'"
).resample(resampling_period).mean()
mango_marker_size = processed.query(
    "Question == @question & TeamName == 'Mango'"
).resample(resampling_period).count().Forecaster
print(mango_marker_size)
print(processed.query("Question == @question & TeamName == 'Mango'"))
fig.add_trace(go.Scatter(
                x=kiwi.index,
                y=kiwi[scoring_method],
                name="Kiwis",
                mode="markers",
                text=[f"Number of forecasts in {resampling_period}: {count}" for count in kiwi_marker_size],
                line_color='deepskyblue',
                marker=dict(
                    size=kiwi_marker_size,
                    sizemode='area',
                    sizeref=2.*max(kiwi_marker_size)/(40.**2),
                    sizemin=4
                ),
                opacity=0.8))

fig.add_trace(go.Scatter(
                x=mango.index,
                y=mango[scoring_method],
                name="Mangoes",
                mode="markers",
                text=[f"Number of forecasts in {resampling_period}: {count}" for count in mango_marker_size],
                line_color='dimgray',
                marker=dict(
                    size=mango_marker_size,
                    sizemode='area',
                    sizeref=2.*max(mango_marker_size)/(40.**2),
                    sizemin=4
                ),
                opacity=0.8))

# Mark 'interventions' or special events
# fig.update_layout(
    # shapes=[
        # # 1st highlight during Feb 4 - Feb 6
        # go.layout.Shape(
            # type="rect",
            # # x-reference is assigned to the x-values
            # xref="x",
            # # y-reference is assigned to the plot paper [0,1]
            # yref="paper",
            # x0="2019-11-15",
            # y0=0,
            # x1="2019-11-16",
            # y1=1,
            # fillcolor="LightSalmon",
            # opacity=0.5,
            # layer="below",
            # line_width=0,
        # ),
        # # 2nd highlight during Feb 20 - Feb 23
        # go.layout.Shape(
            # type="rect",
            # xref="x",
            # yref="paper",
            # x0="2019-11-25",
            # y0=0,
            # x1="2019-11-26",
            # y1=1,
            # fillcolor="LightSalmon",
            # opacity=0.5,
            # layer="below",
            # line_width=0,
        # )
    # ]
# )

fig.update_layout(
    title_text=question,
    legend= {'itemsizing': 'constant'},
)
st.plotly_chart(fig)

show_data = st.checkbox('Show raw data')
if show_data:
    st.subheader('Kiwis')
    kiwi
    st.subheader('Mangoes')
    mango

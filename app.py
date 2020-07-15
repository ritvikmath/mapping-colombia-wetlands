import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from upsampling_utils import create_upsampled_wetlands_map
import plotly.graph_objects as go
import os
import numpy as np
import pickle

########### Define your variables ######

tabtitle = 'wetlands'
heading = 'Wetlands Confidence'

button_input_style = {'font-size': '16px', 'textAlign': 'right', 'background-color': 'green', 'color': 'white'}

BASELINE_METERS_PER_PIXEL = 120
METERS_PER_MILE = 1609
SAMPLE_SIZE = 10000
NO_DATA_VALUE_DICT = {'elevation': 9999999, 'sentinel_infrared': 65535}


########## Set up the chart

########### Initiate the app
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
app.title=tabtitle

########### Set up the layout

app.layout = html.Div(children=[
    html.H1(heading),
    html.Div(children=[
        html.Label(id='data_source_label', children='Data Source:', style={'font-weight': 'bold'}),
        dcc.Dropdown(
            id='data_source',
            options=[
                {'label': 'NASADEM', 'value': 'elevation-raw'},
                {'label': 'NASADEM - Change', 'value': 'elevation-change'},
                {'label': 'Sentinel (Infrared)', 'value': 'sentinel_infrared-raw'},
                {'label': 'Sentinel (Infrared) - Change', 'value': 'sentinel_infrared-change'}
            ],
            value='elevation-change'
        ),
        html.Br(),
        html.Label(id='wetland_type_label', children='Wetland Type:', style={'font-weight': 'bold'}),
        dcc.Dropdown(
            id='wetland_type',
            options=[
                {'label': 'Mangrove', 'value': 'Mangrove'},
                {'label': 'Wet meadow', 'value': 'Wet meadow'},
                {'label': 'Open Water', 'value': 'Open Water'},
                {'label': 'Marsh', 'value': 'Marsh'},
                {'label': 'Floodswamp', 'value': 'Floodswamp'}
            ],
            value='Mangrove'
        ),
        html.Br(),
        html.Label(id='edge_filter_size_label', children='Edge Filter Size:', style={'font-weight': 'bold'}),
        dcc.Dropdown(
            id='edge_filter_size',
            options=[
                {'label': '0.05 sq. mi.', 'value': '.05'},
                {'label': '0.25 sq. mi.', 'value': '.25'},
                {'label': '1 sq. mi.', 'value': '1'}
            ],
            value='.05'
        ),
        html.Br(),
        html.Label(id='interior_point_thresh_label', children='Interior Point Threshold:', style={'font-weight': 'bold'}),
        dcc.Slider(
            id='interior_point_thresh',
            min=0.05,
            max=0.2,
            step=0.05,
            marks={round(i,2):round(i,2) for i in np.arange(0.05, 0.25, 0.05)},
            value=0.1
        ),
        html.Br(),
        html.Label(id='output_file_name_label', children='Output File Name:', style={'font-weight': 'bold', 'width': '100%'}),
        dcc.Input(
            id="output_file_name",
            type='text',
            placeholder="excluding .tiff",
            size=30
        ),
        html.Br(),
        html.Br(),
        dbc.Button("Apply", id='apply_button', style=button_input_style)
        
    ], className='three columns'),
    html.Div(id='output_hists', children=[
        html.Div(id='wetland_hist', children=None),
        html.Div(id='non_wetland_hist', children=None)
        ], className='eight columns', style={'maxWidth': '1000px', 'font-weight': 'bold'})
]
)

########## Define Callback

@app.callback([Output('wetland_hist', 'children'),
               Output('non_wetland_hist', 'children')],
              [Input('apply_button', 'n_clicks')],
              [State('data_source', 'value'),
               State('wetland_type', 'value'),
               State('edge_filter_size', 'value'),
               State('interior_point_thresh', 'value'),
               State('output_file_name', 'value')])
def create_upsampled_map_and_histograms(n_clicks, data_source, wetland_type, edge_filter_size, interior_point_thresh, output_file_name):
    
    #get the current callback context    
    ctx = dash.callback_context

    #if nothing changed, just return
    if not ctx.triggered:
        return None, None
    
    #translate input variables to program-ready variables
    metric, change_status = data_source.split('-')
    change_status = (change_status == 'change')
    
    edge_filter_size = round(float(edge_filter_size)**.5 * METERS_PER_MILE / BASELINE_METERS_PER_PIXEL)
    
    output_file_name = 'assets/%s.tiff'%output_file_name
    
    no_data_value = NO_DATA_VALUE_DICT[metric]
    
    pickle_file_name = '%s_%s_%s_%s'%(data_source, wetland_type, edge_filter_size, interior_point_thresh)
    pickle_file_name = pickle_file_name.replace(' ','')
    interior_pickle_file_name = "%s_interior.p"%pickle_file_name
    exterior_pickle_file_name = "%s_exterior.p"%pickle_file_name
    
    #check if this file has already been created
    if interior_pickle_file_name not in os.listdir('assets') or exterior_pickle_file_name not in os.listdir('assets'):  
        create_upsampled_wetlands_map(metric, change_status, wetland_type, no_data_value, edge_filter_size, interior_point_thresh, output_file_name, 'assets/%s'%pickle_file_name)
     
    print('-------------------------------------------------')
    print('Getting Interior / Exterior Data From Pickle...')
    print('-------------------------------------------------')
    interior_pickle_file_name = 'assets/%s'%interior_pickle_file_name
    exterior_pickle_file_name = 'assets/%s'%exterior_pickle_file_name
    
    interior_vals = pickle.load(open(interior_pickle_file_name, "rb" ))
    exterior_vals = pickle.load(open(exterior_pickle_file_name, "rb" ))
    
    interior_vals = np.random.choice(interior_vals, SAMPLE_SIZE)
    exterior_vals = np.random.choice(exterior_vals, SAMPLE_SIZE)
    
    print('-------------------------------------------------')
    print('Generating Wetlands Histogram...')
    print('-------------------------------------------------')
    fig_wetlands_hist = go.Figure(data=[go.Histogram(x=interior_vals, histnorm='probability density', nbinsx=10)])
    fig_wetlands_hist.update_layout(
        title_text='%s %s distribution'%(wetland_type, data_source),
        xaxis_title_text=data_source, # xaxis label
        yaxis_title_text='Density', # yaxis label
        height=275
    )
    
    print('-------------------------------------------------')
    print('Generating Non-Wetlands Histogram...')
    print('-------------------------------------------------')
    fig_non_wetlands_hist = go.Figure(data=[go.Histogram(x=exterior_vals, histnorm='probability density', nbinsx=10)])
    fig_non_wetlands_hist.update_layout(
        title_text='Non-%s %s distribution'%(wetland_type, data_source),
        xaxis_title_text=data_source, # xaxis label
        yaxis_title_text='Density', # yaxis label
        height=275
    )
    
    print('-------------------------------------------------')
    print('Returning Histograms to App...')
    print('-------------------------------------------------')
    return dcc.Graph(figure=fig_wetlands_hist), dcc.Graph(figure=fig_non_wetlands_hist)

############ Deploy
if __name__ == '__main__':
    app.run_server()
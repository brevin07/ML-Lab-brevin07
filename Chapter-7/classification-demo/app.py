
import numpy as np
from dash import Dash, dcc, html, Input, Output, State
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sklearn.svm import SVC
from dash import dash_table

import utils.dash_reusable_components as drc
import utils.figures as figs
# Import custom reusable components and figure-generation utilities

app = Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1.0"}],
)
app.title = "Breast Cancer SVM Explorer"
server = app.server


def generate_data(n_samples, dataset, noise):
    if dataset == "moons":
        return datasets.make_moons(n_samples=n_samples, noise=noise, random_state=0)
    elif dataset == "circles":
        return datasets.make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=1)
    elif dataset == "linear":
        X, y = datasets.make_classification(
            n_samples=n_samples,
            n_features=2,
            n_redundant=0,
            n_informative=2,
            random_state=2,
            n_clusters_per_class=1,
        )
        rng = np.random.RandomState(2)
        X += noise * rng.uniform(size=X.shape)
        return (X, y)
    elif dataset == "breast_cancer":
        # Load the Breast Cancer dataset from scikit-learn
        data = datasets.load_breast_cancer()
        X = data.data  # Many features available
        y = data.target

        # Use PCA to reduce to 2 dimensions so we can visualize the decision boundary
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        X_reduced = pca.fit_transform(X)

        # Optionally, allow a subset of the full dataset using the sample size slider
        if n_samples < X_reduced.shape[0]:
            idx = np.random.choice(X_reduced.shape[0], n_samples, replace=False)
            X_reduced = X_reduced[idx]
            y = y[idx]

        return X_reduced, y
    else:
        raise ValueError("Data type incorrectly specified. Please choose an existing dataset.")


app.layout = html.Div(
    children=[
        html.Div(
            className="banner",
            children=[
                html.Div(
                    className="container scalable",
                    children=[
                        html.H2(
                            id="banner-title",
                            children=[
                                html.A(
                                    "Breast Cancer SVM Explorer",
                                    href="https://github.com/plotly/dash-svm",
                                    style={"text-decoration": "none", "color": "inherit"},
                                )
                            ],
                        ),
                        html.A(
                            id="banner-logo",
                            children=[html.Img(src=app.get_asset_url("dash-logo-new.png"))],
                            href="https://plot.ly/products/dash/",
                        ),
                    ],
                )
            ],
        ),
        html.Div(
            id="body",
            className="container scalable",
            children=[
                html.Div(
                    id="app-container",
                    children=[
                        html.Div(
                            id="left-column",
                            children=[
                                drc.Card(
                                    id="first-card",
                                    children=[
                                        # Updated dropdown: now includes Breast Cancer
                                        drc.NamedDropdown(
                                            name="Select Dataset",
                                            id="dropdown-select-dataset",
                                            options=[
                                                {"label": "Moons", "value": "moons"},
                                                {"label": "Linearly Separable", "value": "linear"},
                                                {"label": "Circles", "value": "circles"},
                                                {"label": "Breast Cancer", "value": "breast_cancer"},
                                            ],
                                            clearable=False,
                                            searchable=False,
                                            value="breast_cancer",  # Default to breast cancer
                                        ),
                                        drc.NamedSlider(
                                            name="Sample Size",
                                            id="slider-dataset-sample-size",
                                            min=100,
                                            max=500,
                                            step=100,
                                            marks={str(i): str(i) for i in [100, 200, 300, 400, 500]},
                                            value=300,
                                        ),
                                        drc.NamedSlider(
                                            name="Noise Level",
                                            id="slider-dataset-noise-level",
                                            min=0,
                                            max=1,
                                            marks={i / 10: str(i / 10) for i in range(0, 11, 2)},
                                            step=0.1,
                                            value=0.0,  # For real data, noise is set to 0
                                        ),
                                    ],
                                ),
                                drc.Card(
                                    id="button-card",
                                    children=[
                                        drc.NamedSlider(
                                            name="Threshold",
                                            id="slider-threshold",
                                            min=0,
                                            max=1,
                                            value=0.5,
                                            step=0.01,
                                        ),
                                        html.Button("Reset Threshold", id="button-zero-threshold"),
                                    ],
                                ),
                                drc.Card(
                                    id="last-card",
                                    children=[
                                        drc.NamedDropdown(
                                            name="Kernel",
                                            id="dropdown-svm-parameter-kernel",
                                            options=[
                                                {"label": "Radial basis function (RBF)", "value": "rbf"},
                                                {"label": "Linear", "value": "linear"},
                                                {"label": "Polynomial", "value": "poly"},
                                                {"label": "Sigmoid", "value": "sigmoid"},
                                            ],
                                            value="rbf",
                                            clearable=False,
                                            searchable=False,
                                        ),
                                        drc.NamedSlider(
                                            name="Cost (C)",
                                            id="slider-svm-parameter-C-power",
                                            min=-2,
                                            max=4,
                                            value=0,
                                            marks={i: "{}".format(10 ** i) for i in range(-2, 5)},
                                        ),
                                        drc.FormattedSlider(
                                            id="slider-svm-parameter-C-coef",
                                            min=1,
                                            max=9,
                                            value=1,
                                        ),
                                        drc.NamedSlider(
                                            name="Degree",
                                            id="slider-svm-parameter-degree",
                                            min=2,
                                            max=10,
                                            value=3,
                                            step=1,
                                            marks={str(i): str(i) for i in range(2, 11, 2)},
                                        ),
                                        drc.NamedSlider(
                                            name="Gamma",
                                            id="slider-svm-parameter-gamma-power",
                                            min=-5,
                                            max=0,
                                            value=-1,
                                            marks={i: "{}".format(10 ** i) for i in range(-5, 1)},
                                        ),
                                        drc.FormattedSlider(
                                            id="slider-svm-parameter-gamma-coef",
                                            min=1,
                                            max=9,
                                            value=5,
                                        ),
                                        html.Div(
                                            id="shrinking-container",
                                            children=[
                                                html.P("Shrinking"),
                                                dcc.RadioItems(
                                                    id="radio-svm-parameter-shrinking",
                                                    labelStyle={
                                                        "margin-right": "7px",
                                                        "display": "inline-block",
                                                    },
                                                    options=[
                                                        {"label": " Enabled", "value": "True"},
                                                        {"label": " Disabled", "value": "False"},
                                                    ],
                                                    value="True",
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        html.Div(
                            id="div-graphs",
                            children=dcc.Graph(
                                id="graph-sklearn-svm",
                                figure=dict(layout=dict(plot_bgcolor="#282b38", paper_bgcolor="#282b38")),
                            ),
                        ),
                    ],
                )
            ],
        ),
    ]
)


@app.callback(
    Output("slider-svm-parameter-gamma-coef", "marks"),
    [Input("slider-svm-parameter-gamma-power", "value")],
)
def update_slider_svm_parameter_gamma_coef(power):
    scale = 10 ** power
    return {i: str(round(i * scale, 8)) for i in range(1, 10, 2)}


@app.callback(
    Output("slider-svm-parameter-C-coef", "marks"),
    [Input("slider-svm-parameter-C-power", "value")],
)
def update_slider_svm_parameter_C_coef(power):
    scale = 10 ** power
    return {i: str(round(i * scale, 8)) for i in range(1, 10, 2)}


@app.callback(
    Output("slider-threshold", "value"),
    [Input("button-zero-threshold", "n_clicks")],
    [State("graph-sklearn-svm", "figure")],
)
def reset_threshold_center(n_clicks, figure):
    if n_clicks and figure and "data" in figure and len(figure["data"]) > 0:
        try:
            z_data = figure["data"][0].get("z", None)
            # If the 'z' data is a dict, we cannot perform numerical operations; return default.
            if isinstance(z_data, dict):
                return 0.5
            # Convert the retrieved z data to a NumPy array of floats.
            z_array = np.array(z_data, dtype=float)
            # If the array is empty, return default.
            if z_array.size == 0:
                return 0.5
            return -z_array.min() / (z_array.max() - z_array.min())
        except Exception as e:
            print("Error computing threshold from Z data:", e)
            return 0.5
    else:
        return 0.5



@app.callback(
    Output("slider-svm-parameter-degree", "disabled"),
    [Input("dropdown-svm-parameter-kernel", "value")],
)
def disable_slider_param_degree(kernel):
    return kernel != "poly"


@app.callback(
    Output("slider-svm-parameter-gamma-coef", "disabled"),
    [Input("dropdown-svm-parameter-kernel", "value")],
)
def disable_slider_param_gamma_coef(kernel):
    return kernel not in ["rbf", "poly", "sigmoid"]


@app.callback(
    Output("slider-svm-parameter-gamma-power", "disabled"),
    [Input("dropdown-svm-parameter-kernel", "value")],
)
def disable_slider_param_gamma_power(kernel):
    return kernel not in ["rbf", "poly", "sigmoid"]


@app.callback(
    Output("div-graphs", "children"),
    [
        Input("dropdown-svm-parameter-kernel", "value"),
        Input("slider-svm-parameter-degree", "value"),
        Input("slider-svm-parameter-C-coef", "value"),
        Input("slider-svm-parameter-C-power", "value"),
        Input("slider-svm-parameter-gamma-coef", "value"),
        Input("slider-svm-parameter-gamma-power", "value"),
        Input("dropdown-select-dataset", "value"),
        Input("slider-dataset-noise-level", "value"),
        Input("radio-svm-parameter-shrinking", "value"),
        Input("slider-threshold", "value"),
        Input("slider-dataset-sample-size", "value"),
    ],
)
def update_svm_graph(
    kernel,
    degree,
    C_coef,
    C_power,
    gamma_coef,
    gamma_power,
    dataset,
    noise,
    shrinking,
    threshold,
    sample_size,
):
    # ... [existing code to process data and train model] ...
    X, y = generate_data(n_samples=sample_size, dataset=dataset, noise=noise)
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

    x_min = X[:, 0].min() - 0.5
    x_max = X[:, 0].max() + 0.5
    y_min = X[:, 1].min() - 0.5
    y_max = X[:, 1].max() + 0.5
    h = 0.3  # mesh step size
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    C = C_coef * 10 ** C_power
    gamma = gamma_coef * 10 ** gamma_power
    flag = True if shrinking == "True" else False

    # Train the SVM model
    clf = SVC(C=C, kernel=kernel, degree=degree, gamma=gamma, shrinking=flag)
    clf.fit(X_train, y_train)

    # Get decision boundary values to support plotting (as before)
    if hasattr(clf, "decision_function"):
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

    # Get plot figures for prediction and ROC curve as before
    prediction_figure = figs.serve_prediction_plot(
        model=clf,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        Z=Z,
        xx=xx,
        yy=yy,
        mesh_step=h,
        threshold=threshold,
    )
    roc_figure = figs.serve_roc_curve(model=clf, X_test=X_test, y_test=y_test)

    # Get the confusion matrix table data and column definitions from our helper
    table_data, table_columns = figs.serve_confusion_matrix_table(
        model=clf, X_test=X_test, y_test=y_test, Z=Z, threshold=threshold
    )

    # Return the components – here, we replace the pie chart with a DataTable
    return [
        html.Div(
            id="svm-graph-container",
            children=dcc.Loading(
                className="graph-wrapper",
                children=dcc.Graph(id="graph-sklearn-svm", figure=prediction_figure),
                style={"display": "none"},
            ),
        ),
        html.Div(
            id="graphs-container",
            children=[
                dcc.Loading(
                    className="graph-wrapper",
                    children=dcc.Graph(id="graph-line-roc-curve", figure=roc_figure),
                ),
                dcc.Loading(
                    className="graph-wrapper",
                    children=dash_table.DataTable(
                        id="table-confusion-matrix",
                        columns=table_columns,
                        data=table_data,
                        style_cell={'textAlign': 'center', 'padding': '5px'},
                        style_header={'fontWeight': 'bold'},
                    ),
                ),
            ],
        ),
    ]



if __name__ == "__main__":
    app.run(debug=True)

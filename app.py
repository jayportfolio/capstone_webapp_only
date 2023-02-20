import time

import pandas as pd
import streamlit as st
import numpy as np
import random
import matplotlib.pyplot as plt
import os
from PIL import Image

from sklearn.metrics import PredictionErrorDisplay
from sklearn.model_selection import cross_validate, cross_val_predict

from utility_functions.functions_b__get_the_data import get_source_dataframe
from utility_functions.functions_d3__prepare_store_data import this_test_data
from utility_functions.functions_gh_presentation_and_launch import load_model, get_webapp_models

RAND_INDEX_CSV = "webapp_deployment/cache/rand_index.csv"
# RANDOM_INSTANCE_CSV = "random_instance.csv"
RANDOM_INSTANCE_PLUS_CSV = "webapp_deployment/cache/random_instance_plus.csv"

historic_predicted_prices = []

st.set_option('deprecation.showfileUploaderEncoding', False)

df, X_test, y_test, feature_names = None, None, None, None
rand_index = -1

DATA_VERSION = None
previous_data_version = DATA_VERSION

# prediction_models = {
#     'XG Boost (data version 11) - Best model': 'optimised_model_XG Boost (tree)_v11',
#     'Stacked Model [xgb,lgb,knn] (data version 6) - Great model': 'optimised_model_Stacked Model_v06',
#     'KNN (data version 6) - Fastest to train, Good model': 'optimised_model_KNN_v06',
#     'XG Boost (data version 10) - Good model': 'optimised_model_XG Boost (tree)_v10',
#     'Catboost (data version 6) - Good model': 'optimised_model_CatBoost_v06',
#     'Light Gradient Boosting (data version 6) - Good model': 'optimised_model_Light Gradient Boosting_v06',
#     'Stacked Model - still in beta (data version 11)': 'optimised_model_Stacked Model_v11',
#     'Random Forests (data version 9) - Fair model': 'optimised_model_Random Forest_v09',
#     'Neural Network (data version 11) - Mediocre model': 'optimised_neural network m16 mega + dropout (v11)_v11',
#     'Linear Regression (data version 11) - Poor model': 'optimised_model_Linear Regression (Ridge)_v11',
#     'Linear Regression (data version 6) - Poor model': 'optimised_model_Linear Regression (Ridge)_v06',
# }

prediction_models = get_webapp_models()
if not prediction_models:
    raise AttributeError("there are no models available! You need to populate the model_list/webapp_final_models directory with some models trained on the data.")


def main():
    global X_test, y_test, feature_names, rand_index, DATA_VERSION, previous_data_version

    st.markdown(
        "<h2 style='text-align: center; color: White;background-color:#e84343'>London Property Prices Predictor</h2>",
        unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center; color: Black;'>Choose your algorithm and feature set to make property price predictions</h5>",
                unsafe_allow_html=True)

    st.sidebar.header("")
    # st.sidebar.markdown(
    #     "This is a Web app that can predict the price of a London property based on features of that property.")
    # st.sidebar.header("Sidebar Options")

    available_model_names = prediction_models.keys()

    # def print_something(filter_versions):
    #    print('something')
    # date = st.selectbox('Select a day', dates, format_func=day_name, key='date', index=st.session_state['date_index'], on_change=update_date_index, args=(dates,))
    filter_versions = ['all versions', 'v06', 'v09', 'v10', 'v11']

    # limit_to_version = st.selectbox('Limit to version?', filter_versions, on_change=(print_something), key='date', args=(filter_versions,))

    def model_changed():
        # predict_and_display(manual_parameters, model, selected_model, selected_model_key, selected_model_name, selected_model_version, test_size, update_property=True,
        #                    updateable=container_col1_data)
        # pass

        global feature_names, rand_index, X_test, y_test
        rand_index, predict_instance, single_real_price = get_random_instance(X_test, y_test)
        fix_to = rand_index if rand_index in [1435, 3346] else ""
        update_about_property(feature_names, rand_index, predict_instance, fix_to=fix_to)

    selected_model_key = st.selectbox('Which model do you want to use?', available_model_names, on_change=model_changed)

    print("prediction_models", prediction_models)
    print("selected_model_key", selected_model_key)
    selected_model_name, selected_model_verdict, selected_model, selected_model_version = prediction_models[selected_model_key]
    print("selected_model", selected_model)
    print("selected_model_name", selected_model_name)
    print("selected_model_verdict", selected_model_verdict)
    print("selected_model_version", selected_model_version)
    model = load_model_wrapper(selected_model, model_type='neural' if 'eural' in selected_model_key else 'standard')

    load_message = 'Please wait while data loads    ...'
    col1, col2, col3 = st.tabs(["Single Prediction", "Multiple Predictions", "Additional Detail"])

    with col1:

        with st.spinner(load_message):
            container_col1_header = st.container()
            container_col1_data = st.container()
            container_col1_footnote = st.container()

            # manual_parameters = st.checkbox('Use manual parameters instead of sample')
            manual_parameters = False
            if not manual_parameters:
                DATA_VERSION = selected_model_version
                print("DATA_VERSION", DATA_VERSION)
                X_test, y_test, feature_names = this_test_data(VERSION=DATA_VERSION, test_data_only=True, cloud_or_webapp_run=True, versioned=True)
                test_size = len(y_test)
                # pass

            else:
                lati = st.slider("Input Your latitude", 51.00, 52.00)
                longi = st.slider("Input your longitude", -0.5, 0.3)
                beds = st.slider("Input number of bedrooms", 0, 6)
                baths = st.slider("Input number of bathrooms", 0, 6)

                inputs = [[lati, longi, beds, baths]]

            # if st.sidebar.button('Change the random property!'):
            #    test_size = change_the_random_property(DATA_VERSION, test_size)

            if container_col1_header.button('Choose another random property, and predict'):
                test_size = change_the_random_property(DATA_VERSION, test_size, updatables=[container_col1_footnote])
                model = predict_and_display(manual_parameters, model, selected_model, selected_model_key, selected_model_name, selected_model_version, test_size,
                                            update_property=False, updateable=container_col1_data)

            if container_col1_header.button('Use example property 1'):
                param="1435"
                print("param", param)
                test_size = change_the_random_property(DATA_VERSION, test_size, fix_to=param, updatables=[container_col1_footnote])
                model = predict_and_display(manual_parameters, model, selected_model, selected_model_key, selected_model_name, selected_model_version, test_size,
                                            update_property=False, updateable=container_col1_data)

            if container_col1_header.button('Use example property 2'):
                param="3346"
                test_size = change_the_random_property(DATA_VERSION, test_size, fix_to=param, updatables=[container_col1_footnote])
                model = predict_and_display(manual_parameters, model, selected_model, selected_model_key, selected_model_name, selected_model_version, test_size,
                                            update_property=False, updateable=container_col1_data)

            if container_col1_header.button('Predict again for the same property'):
                model = predict_and_display(manual_parameters, model, selected_model, selected_model_key, selected_model_name, selected_model_version, test_size,
                                            update_property=False, updateable=container_col1_data)
                rand_index, predict_instance, single_real_price = get_random_instance(X_test, y_test)
                fix_to = rand_index if rand_index in [1435,3346] else ""
                update_about_property(feature_names, rand_index, predict_instance, fix_to=fix_to)

    with col2:

        with st.spinner(load_message):

            if st.checkbox('View all available predictions (entire test set)'):
                DATA_VERSION = selected_model[-2:]
                DATA_VERSION = selected_model_version

                X_test, y_test, feature_names = this_test_data(VERSION=DATA_VERSION, test_data_only=True, cloud_or_webapp_run=True, versioned=True)
                try:
                    acc = model.score(X_test, y_test)
                    st.write('Accuracy of test set: ', acc)
                except:
                    pass

                y_pred = model.predict(X_test).flatten()
                multiple_predictions = np.vstack((y_test.flatten(), y_pred)).T
                multiple_predictions_df = pd.DataFrame(multiple_predictions, columns=['Actual Price', 'Predicted Price'])
                multiple_predictions_df['Actual Price'] = pd.to_numeric(multiple_predictions_df['Actual Price']).astype(int)
                multiple_predictions_df['Predicted Price'] = pd.to_numeric(multiple_predictions_df['Predicted Price']).astype(int)

                st.write(multiple_predictions_df)
                print("type(multiple_predictions_df):", type(multiple_predictions_df))

            if not manual_parameters:
                pass
                # if st.button('Get a different random property!'):
                #     rand_index, random_instance, random_instance[0] = randomise_property(DATA_VERSION, test_size)
                #     st.text(f'sample variables ({rand_index}): {random_instance[0]}')
                #     st.text(f'Expected prediction: {y_test[rand_index]}')

    with col3:
        with st.spinner(load_message):
            X_test, y_test, feature_names = this_test_data(VERSION=DATA_VERSION, test_data_only=True, cloud_or_webapp_run=True, versioned=True)
            try:
                acc = model.score(X_test, y_test)
                st.write('Accuracy of test set: ', acc)
            except:
                pass

            if st.checkbox('Show the underlying dataframe'):
                DATA_VERSION = selected_model[-2:]
                DATA_VERSION = selected_model_version

                df, df_type = get_source_dataframe(cloud_or_webapp_run=True, version=DATA_VERSION, folder_prefix='')
                print("claiming to be colab so I can use the cloud version of data and save space")
                st.write(df)


def predict_and_display(manual_parameters, model, selected_model, selected_model_key, selected_model_name, selected_model_version, test_size, update_property=True, updateable=st, fix_to=""):
    global DATA_VERSION, X_test, y_test, feature_names, rand_index, previous_data_version, historic_predicted_prices
    DATA_VERSION = selected_model[-2:]
    DATA_VERSION = selected_model_version
    X_test, y_test, feature_names = this_test_data(VERSION=DATA_VERSION, test_data_only=True, cloud_or_webapp_run=True, versioned=True)
    try:
        acc = model.score(X_test, y_test)
        updateable.write('Accuracy of test set: ', acc)
    except:
        pass

    if not manual_parameters:
        rand_index, predict_instance, single_real_price = get_random_instance(X_test, y_test)

        if update_property:
            update_about_property(feature_names, rand_index, predict_instance, fix_to=fix_to)

    model = load_model_wrapper(selected_model, model_type='neural' if 'eural' in selected_model_key else 'standard')
    X_test, y_test, feature_names = this_test_data(VERSION=DATA_VERSION, test_data_only=True, cloud_or_webapp_run=False, versioned=True)

    predict_instance_array = [predict_instance]

    result_arr = model.predict(predict_instance_array)
    flat_result_arr = result_arr.flatten().astype(float)
    single_predicted_price = flat_result_arr[0]

    try:
        print("0 historic_predicted_prices:", st.session_state['historic_predicted_prices'], single_real_price)
    except:
        st.session_state['historic_predicted_prices'] = []
    hpp = st.session_state['historic_predicted_prices']
    hpp.append(single_predicted_price)
    st.session_state['historic_predicted_prices'] = hpp
    print("1 historic_predicted_prices:", st.session_state['historic_predicted_prices'], single_real_price)

    difference = abs(single_real_price - single_predicted_price)
    if difference < 10000:
        remark, colour = 'good', 'limegreen'
    elif difference < 50000:
        remark, colour = 'ok', 'khaki'  # 'yellow'
    else:
        remark, colour = 'poor', 'lightcoral'  # 'red'

    fig, ax = plt.subplots()
    # X_test, y_test, feature_names = this_test_data(VERSION=DATA_VERSION, test_data_only=True, cloud_or_webapp_run=False, versioned=True)
    # model = load_model_wrapper(selected_model, model_type='neural' if 'eural' in selected_model_key else 'standard')
    y_pred = model.predict(X_test).flatten()
    if 'eural' in selected_model_key:
        from sklearn.metrics import r2_score
        st.write('Score:', r2_score(y_test, y_pred))
        try:
            st.write('Accuracy of test set: ', acc)
        except:
            pass
    ax.scatter(y_test, y_pred, s=25, c='silver')
    # ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, c='black')
    # ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], lw=1, c='dimgray')
    ax.plot([100000, 620000], [100000, 620000], lw=1, c='darkgrey')
    ax.set_xlim([90000, 650000])
    ax.set_ylim([50000, 750000])
    try:
        other_historic_predictions = [x for x in st.session_state['historic_predicted_prices'] if x != single_predicted_price]
        for historic_prediction in other_historic_predictions:
            ax.scatter(single_real_price, historic_prediction, s=80, c='dimgrey')
        # for i in range(100000, 600000, 50000):
        #    ax.scatter(i, i, s=100, c='red')
        ax.scatter(single_real_price, single_predicted_price, s=100, c=colour)
    except:
        pass
    ax.set_title("Comparing actual vs predicted property values. \nPredictions by " + selected_model_name)
    ax.set_xlabel('Actual property price')
    ax.set_ylabel('Predicted property price')
    plt.ticklabel_format(style='plain')
    updateable.pyplot(fig)

    updateable.write(
        "Legend: Coloured dot is the current prediction, compared to the real price. Green = good prediction, \n   Yellow=ok prediction, \n   Red=Poor prediction.")
    updateable.write(
        "Dark grey=a prediction you tried for this property earlier.")
    updateable.write(
        "Light grey spectrum of dots=predictions compared to actual prices across the entire test set for the chosen model.")

    updateable.info('The actual price for this property is £{:.0f}'.format(single_real_price))
    if remark == 'good':
        updateable.success('The predicted price for this property is £{:.0f}'.format(single_predicted_price) + '\n')
    elif remark == 'ok':
        updateable.warning('The predicted price for this property is £{:.0f}'.format(single_predicted_price) + '\n')
    else:
        updateable.error('The predicted price for this property is £{:.0f}'.format(single_predicted_price) + '\n')

    return model


def change_the_random_property(DATA_VERSION, test_size, fix_to="", updatables=[]):
    print("|| 1:", "fix_to:", fix_to)
    X_test, y_test, feature_names = this_test_data(VERSION=DATA_VERSION, test_data_only=True, cloud_or_webapp_run=True, versioned=True)
    test_size = len(y_test)
    # rand_index, random_instance, random_instance[0] = randomise_property(DATA_VERSION, test_size)
    rand_index, predict_instance, expected = randomise_property(DATA_VERSION, test_size, fix_to=fix_to)
    # print("rand_index", rand_index)
    # print("random_instance", random_instance)
    # print("random_instance[0]", len(random_instance[0]))
    update_about_property(feature_names, rand_index, predict_instance, updatables=updatables, fix_to=fix_to)
    return test_size


def update_about_property(feature_names, rand_index, predict_instance, updatables=[], fix_to=""):
    load_message2 = "Loading property information..."
    with st.spinner(load_message2):
        st.sidebar.subheader("About your chosen property")
        # df, df_type = get_source_dataframe(cloud_or_webapp_run=True, version=DATA_VERSION, folder_prefix='')
        # print("rand_index", rand_index)
        # print("random_instance", predict_instance)
        # #print("random_instance[0]", len(random_instance[0]))
        # print("feature_names", len(feature_names), feature_names[0], '...', feature_names[-1])
        random_instance_df = pd.DataFrame(predict_instance, index=feature_names)
        random_instance_df.columns = ['random=' + str(rand_index)]

        print(random_instance_df)
        bedrooms = int(random_instance_df.loc[['bedrooms']].values[0][0])
        property_type_enum = random_instance_df.loc[['tenure.tenureType_LEASEHOLD', 'tenure.tenureType_FREEHOLD']]

        property_type = 'flat'
        print(f"bedrooms: >{bedrooms}<")
        print(f"property_type:, >{property_type}<")

        print("|| 24:", "fix_to:", fix_to)
        image = get_image_for_property(bedrooms, property_type, str(fix_to))

        #st.sidebar.image(image, caption='Image of two bedroom flat')
        st.sidebar.image(image)

        st.sidebar.table(random_instance_df)
        for each in updatables:
            each.table(random_instance_df)


@st.cache_data
def get_image_for_property(bedrooms, property_type, fix_to: str = ""):
    print("||get_image_for_property")
    if fix_to not in ["",None]:  # fix_to:
        try:
            filename = f'webapp_deployment/webapp_media/example_property_{fix_to}.png'
            print(f'try to find: >>{filename}<<')
            image = Image.open(filename)
            print(f'Found: >>{filename}<<')
            return image
        except:
            image = Image.open('webapp_deployment/webapp_media/image_not_available.png')
            print(f'defaulting to n/a image')
            return image

    try:
        # print("||get_image_for_property")
        import random
        # print('||imported random')
        rand = random.randint(0, 9)
        # print('|| rand:', rand)
        filename = f'webapp_deployment/webapp_media/{bedrooms}_bedroom_{property_type}_{rand}.png'
        # print(f'try to find: >>{filename}<<')

        image = Image.open(filename)
        print(f'Found: >>{filename}<<')
        return image
    except Exception as e:
        print(e)
        pass

    try:
        # xxximage = Image.open('webapp_media/2_bedroom_flat.png')
        filename = f'webapp_deployment/webapp_media/{bedrooms}_bedroom_{property_type}.png'
        print(f'try to find: >>{filename}<<')
        image = Image.open(filename)
        print(f'Found: >>{filename}<<')
        return image

    except:
        image = Image.open('webapp_deployment/webapp_media/image_not_available.png')
        print(f'defaulting to n/a image')
        return image


def randomise_property(DATA_VERSION, test_size, fix_to=None):
    print("|| 3:", "fix_to:", fix_to)
    st.session_state['historic_predicted_prices'] = []
    if fix_to:
        rand_index = int(fix_to)
    else:
        rand_index = random.randint(0, test_size - 1)

    X_test, y_test, feature_names = this_test_data(VERSION=DATA_VERSION, test_data_only=True, cloud_or_webapp_run=True, versioned=True)

    random_instance = [X_test[rand_index]]
    # np.savetxt(RANDOM_INSTANCE_CSV, random_instance, delimiter=",")
    expected = y_test[rand_index]
    # np.savetxt(RANDOM_INSTANCE_CSV, random_instance, delimiter=",")
    random_instance_plus = [rand_index, expected]
    random_instance_plus.extend(random_instance[0])
    print("random_instance_plus:", random_instance_plus)
    np.savetxt(RANDOM_INSTANCE_PLUS_CSV, [random_instance_plus], delimiter=",")
    np.savetxt(RAND_INDEX_CSV, [rand_index], delimiter=",")

    # return rand_index, random_instance, random_instance[0]
    return get_random_instance(X_test, y_test)


def get_random_instance(X_test, y_test):
    try:
        raise InterruptedError("don't ever do this actually")
        random_instance_plus = np.loadtxt(RANDOM_INSTANCE_PLUS_CSV, delimiter=",")
        print("random_instance_plus:", random_instance_plus)
        rand_index = int(random_instance_plus[0])
        print("rand_index:", rand_index)
        expected = random_instance_plus[1]
        print("expected:", expected)
        inputs = [random_instance_plus[2:]]
        print("[random_instance_plus[2:]]:", [random_instance_plus[2:]])

        random_instance = [X_test[rand_index]]

    except:
        try:
            print('start')
            print("trying to get old rand_index")
            rand_index_arr = np.loadtxt(RAND_INDEX_CSV, delimiter=",")
            print("found old rand_index", rand_index_arr)
            rand_index = int(rand_index_arr)
            print("loaded old rand_index of", rand_index)
        except:
            print("couldn't retrieve the old rand_index, generating a new one")
            test_size = len(X_test)
            rand_index = random.randint(0, test_size - 1)
            print("new rand_index is", rand_index)
            np.savetxt(RAND_INDEX_CSV, [rand_index], delimiter=",")
            print("saved new rand_index of", rand_index)

    random_instance = [X_test[rand_index]]
    inputs = random_instance
    expected = y_test[rand_index]

    random_instance_plus = [rand_index, expected]
    random_instance_plus.extend(random_instance[0])
    np.savetxt(RANDOM_INSTANCE_PLUS_CSV, random_instance_plus, delimiter=",")
    # return random_instance, expected
    # return [random_instance_plus[2:]], expected

    predict_instance = random_instance_plus[2:]
    return rand_index, predict_instance, expected


def load_model_wrapper(selected_model, model_type):
    return load_model(selected_model, model_type=model_type)


if __name__ == '__main__':
    main()

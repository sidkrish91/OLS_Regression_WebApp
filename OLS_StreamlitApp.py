# -*- coding: utf-8 -*-
"""
Version 2a
changelog from 1.1c:
1) This version uses only Ordinary Least Squares to perform  regression
2) The GUI is changed to accomadate different pages that enable the different stages of model analysis
    a) Page 0: Loading the csv data onto a pandas DataFrame. 
    b) Page 1: Data visualisation using line plots, scatter plots and scatter_matrix
    c) page 3: Model building. Here linear least squares model is built. User can understand the model performance.
    the model is built on float32 dtypes. this should reduce the computation time
3) Use of altair charts for visualisation. Altair comes packaged with streamlit. Dependency on Matplolib is removed.

    @author: Siddharth K
"""

# Importing the required libraries. 
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn import linear_model as lm
import sklearn.metrics as metrics
import altair as alt


st.title("OLS Regression application") 


def main():
    global df, cols
    
    file_upload = st.sidebar.file_uploader('Browse for csv file',type=['csv'])
    data_type = st.sidebar.radio('Index dtype',("TimeStamp","Numeric"))
    
    if file_upload is not None:
        df,col_names = read_data(file_upload,data_type)
    else:
        st.write('Please upload a CSV data')
        st.stop()

    pages = {
        "Visualisation": viz_plot,
        "Regression Modeling":reg_model
    }

    page = st.sidebar.radio("Select your page", tuple(pages.keys()))

    pages[page](bool)

    

def read_data(loc,index_type):
    global df, cols
    
    try:
        
        if index_type =='TimeStamp':
            df = pd.read_csv(loc,header='infer',index_col=0)
            df.index = pd.to_datetime(df.index)
            cols = list(df.columns)
            df = df[cols].apply(pd.to_numeric,errors='coerce',axis=1)
            df= df.dropna(axis=1, how='all')
            df= df.dropna(axis=0, how='any')
            cols = list(df.columns)
        
            
        else: #if Numeric==True, index is created by Pandas
            df = pd.read_csv(loc,header='infer')
            cols = list(df.columns)
            df = df[cols].apply(pd.to_numeric,errors='coerce',axis=1)
            df= df.dropna(axis=1, how='all')
            df= df.dropna(axis=0, how='any')
            cols = list(df.columns)
            
    except Exception as e:
         st.write(e)
         st.stop()
         
    return df,cols



def reg_model(bool):
    global df, cols
    
    try:
#        model_type = st.sidebar.radio('Pick model type',('OLS'))
        X_list = st.sidebar.multiselect('Select predictors',cols)
        poly_deg = st.sidebar.slider('Polynomial degree',min_value=1,max_value=3,step=1)
        Y_list = st.sidebar.multiselect('Select target',cols)
        scaler = st.sidebar.checkbox('Standardize',value=False)
        
        
        if len(X_list)==0 or len(Y_list)==0:
            st.write('Select X (predictor) and / or Y (response) variable(s)')
        elif len(Y_list)>1:
            st.write('Too many response variables chosen. Select only one')
        else:
            check = any(item in X_list for item in Y_list)
            if check:
                st.write('Common variables present in X and Y. This will raise errors in model building')
                st.stop()
            p = PolynomialFeatures(degree = poly_deg,include_bias=False).fit(df[X_list])
            df_poly = pd.DataFrame(p.transform(df[X_list]),
                                   columns = p.get_feature_names(df[X_list].columns),index=df.index)
            df_poly = pd.merge(df_poly,df[Y_list],left_index=True,right_index=True)
        
            prediction,coef = ordinary_regression(scaler,df_poly,X_list,Y_list)
            plot_line(df_poly[Y_list].values.flatten(),prediction['Complete'],df_poly.index)
            st.text('Training RMSE:  '+str(prediction['RMSE_train']))
            st.text('Testing RMSE:      '+str(prediction['RMSE_test']))
            st.text('Training R2:    '+str(prediction['R2Train']))
            st.text('Testing R2:    '+str(prediction['R2Test']))
            st.header('Scatter plot: Actual vs Predicted')
            plot_scatter(prediction['Train_Y'].values.flatten(),prediction['Test_Y'].values.flatten(),
                            prediction['Train'],prediction['Test'])
            st.header('Coefficient weights')
            rem_intcpt = st.checkbox('Remove Intercept',value=False)
            plot_coefBar(coef,rem_intcpt)
                
    except Exception as e:
        st.write(e)
        st.stop()
        

def ordinary_regression(norm,DF,xVarName,yVarName):
    try:
       
        X_train,X_test,Y_train,Y_test = train_split(DF,xVarName,yVarName)
        
        if norm:  
            linear_estimator =  [("scaler",StandardScaler()),("OLS",lm.LinearRegression(fit_intercept=False,normalize=False))]
        else:
            linear_estimator =  [("OLS",lm.LinearRegression(normalize=False))]
        linear_model = Pipeline(linear_estimator)
        
       
        try:
            linear_model.fit(X_train,Y_train)
        except Exception as e:
            st.write(e)
            st.stop()
        if linear_model.named_steps['OLS'].rank_ != np.shape(X_train)[1]:
            st.sidebar.write('Collinear variables are present. Use Shrinkage model(s)')
            
        
        pred_train = linear_model.predict(X_train)
        pred_test = linear_model.predict(X_test)
        pred_all = linear_model.predict(DF[xVarName])
        r2_train = metrics.r2_score(Y_train,pred_train)
        r2_test = metrics.r2_score(Y_test,pred_test)
        train_rmse = np.sqrt(metrics.mean_squared_error(Y_train,pred_train))
        test_rmse = np.sqrt(metrics.mean_squared_error(Y_test,pred_test))
        pred_dict = {'Train_Y':Y_train,'Test_Y':Y_test,'Train':pred_train.flatten(),'Test':pred_test.flatten(),'Complete':pred_all.flatten(),
                     'R2Train':r2_train,'R2Test':r2_test,'RMSE_train':train_rmse,'RMSE_test':test_rmse}
    
        coefs = linear_model.named_steps['OLS'].coef_
        
        
        coef_df = pd.DataFrame({'Variables':xVarName,'Coefficient':np.transpose(coefs)[:,0]})
        intcpt = linear_model.named_steps['OLS'].intercept_
       
        if isinstance(intcpt,float):
            coef_df = coef_df.append({'Variables':'Intercept',
                                  'Coefficient':intcpt},ignore_index=True)
        else:
            
            coef_df = coef_df.append({'Variables':'Intercept',
                                  'Coefficient':intcpt[0]},ignore_index=True)
      
    except Exception as e:
        st.write(e)
        st.stop()
        
    return pred_dict,coef_df


def train_split(DF,xVarName,yVarName):
     try:
         
         X_train,X_test,Y_train,Y_test = train_test_split(DF[xVarName],DF[yVarName], test_size=0.15,
                                                          shuffle=False, random_state=42)
         X_train=X_train.astype('float32')
         X_test=X_test.astype('float32')
         Y_train=Y_train.astype('float32')
         Y_test=Y_test.astype('float32')
     except Exception as e:
          st.write(e)
          st.stop()
     return X_train,X_test,Y_train,Y_test
 
    

def viz_plot(bool):
    global df, cols
    try:
        st.subheader('Data - After removal of NAN and Null')
        st.write(df)
        vizType = st.sidebar.radio('Plot type',('Line','Scatter'))
        
        if df.empty:
            st.write('DataFrame is empty. Please check the data for NAN and NULL values. Columns and rows with are removed while uploading')
            st.stop()
#        elif len(X_list)==0:
#            st.write('Please Select variable to plot')
        elif vizType =='Line':
            X_list = st.sidebar.multiselect('Y-axis',cols)
            st.header('Line plot')
            if len(X_list)>1:
                st.write('Selecting more than one variable will result in detrimental plot visuals. BE warned!!' )
            df.index.name="x"
            source = df[X_list]
#            source = source.reset_index().melt('x',var_name='Variable',value_name='y')
#            nearest = alt.selection(type='single',nearest=True,on='mouseover',fields=['x'],empty='none')
#            line = alt.Chart(source).mark_line(interpolate='basis').encode(x='x', y='y',color='Variable:N')
#            selectors = alt.Chart(source).mark_point().encode(x='x',opacity=alt.value(0)).add_selection(nearest)
#            points = line.mark_point().encode(opacity=alt.condition(nearest, alt.value(1), alt.value(0)))
#            text = line.mark_text(align='left', dx=5, dy=-5).encode(text=alt.condition(nearest,'y', alt.value(' ')))
#            rules = alt.Chart(source).mark_rule(color='gray').encode(x='x',).transform_filter(nearest)
#            st.altair_chart(alt.layer(line, selectors, points, rules, text).interactive(),use_container_width=True)
            
            st.altair_chart(alt.Chart(source.reset_index().melt('x')).mark_line().encode(alt.X('x'),alt.Y('value',scale=alt.Scale(zero=False)),color='variable').interactive(),use_container_width=True)
        elif vizType =='Scatter':
            st.header('Pairwise Scatter plot')
            X_list = st.sidebar.multiselect('Paired variable(s)',cols)
            if len(X_list)>2:
                st.write('Please select only 2 variables to generate pariwise scatter plot')
            elif len(X_list)==0 or len(X_list)==1:
                st.write('Please select any 2 variables to plot')
            elif len(X_list)==2:
                source = df[X_list]
                x = source.columns[0]
                y = source.columns[1]
#                st.altair_chart(alt.Chart(source).mark_circle(size=15).encode(x=x,y=y).interactive(),use_container_width=True)
                chart = alt.Chart(source).mark_circle(size=20).encode(x=x,y=y)
                st.altair_chart(chart+chart.transform_regression(x,y,method='linear').mark_line().interactive(),use_container_width=True)
                if st.sidebar.button('Show Scatter Matrix'):
                    st.header ('Scatter Matrix')
                    st.altair_chart(alt.Chart(df).mark_circle().encode(alt.X(alt.repeat("column"), type='quantitative'),
    alt.Y(alt.repeat("row"), type='quantitative')).properties(width=150, height=150).repeat(row=list(df.columns),column=list(df.columns)).interactive(),use_container_width=True)
                    
            
    except Exception as e:
        st.write(e)
        st.stop()

def plot_line(actual,predicted,index):
    try:
        temp_df = pd.DataFrame({'Actual':actual,'Predicted':predicted},index=index)
        temp_df.index.name='x'
    except Exception as e:
        st.write(e)
        st.stop()
        
    return st.altair_chart(alt.Chart(temp_df.reset_index().melt('x')).mark_line().encode(x='x',y='value',color='variable').properties().interactive(),use_container_width=True)


def plot_scatter(actualTrain,actualTest,predTrain,predTest):
    try:
        
        train_df = pd.DataFrame({'Actual':actualTrain,'Prediction':predTrain})
        test_df = pd.DataFrame({'Actual':actualTest,'Prediction':predTest})
        c1 = alt.Chart(train_df).mark_circle(size=20).encode(x='Actual',y='Prediction').properties(title='Training Data')
        train_figure = c1+c1.transform_regression('Actual','Prediction',method='linear').mark_line(color='red')
        c2 = alt.Chart(test_df).mark_circle(size=20).encode(x='Actual',y='Prediction').properties(title='Testing Data')
        test_figure = c2+c2.transform_regression('Actual','Prediction',method='linear').mark_line(color='red').interactive()    
    except Exception as e:
        st.write(e)
        st.stop()
        
    return st.altair_chart(train_figure | test_figure, use_container_width=True)

def plot_coefBar(coef,intcpt_state):
    try:
        if intcpt_state:
            coef = coef[coef['Variables']!='Intercept']
            c=alt.Chart(coef).mark_bar().encode(x='Variables:N',y='Coefficient:Q',
                       color=alt.condition(alt.datum.Coefficient>0,alt.value("steelblue"),alt.value("orange")))
            c_text=c.mark_text(align='left',baseline='middle',dx=5,dy=-10).encode(text=alt.Text('Coefficient',format='.3f'))
            c_chart=(c+c_text).interactive()
        else:
            c=alt.Chart(coef).mark_bar().encode(x='Variables:N',y='Coefficient:Q',
                       color=alt.condition(alt.datum.Coefficient>0,alt.value("steelblue"),alt.value("orange")))
            c_text=c.mark_text(align='center',baseline='line-bottom',dx=5,dy=-10,fontSize = 15).encode(text=alt.Text('Coefficient',format='.3f'))
            c_chart=(c+c_text).interactive()
    
    except Exception as e:
        st.write(e)
        st.stop()
        
    return st.altair_chart(c_chart,use_container_width=True), st.subheader('Coefficient Table'),st.write(coef)


if __name__ == "__main__":
    main()
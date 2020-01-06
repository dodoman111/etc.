'''
###################################################################################################################################
update 2019-03-15
- data 불러올때 랜덤 샘플링 추가

update 2019-03-18
- feature transform function 추가 : *를 이용하여 base과 상관 feature의 polynomial features 형성

update 2019-03-19
- feature transform function 고려 : box-cox 변환...-> data가 0인 값이 존재하기 때문에 쓸 수 없음(boxcox는 데이터가 양수여야 가능).

update 2019-03-20
- 문자 list를 입력받아 data 생성 코드 완성(generate_data)
- 시각화 : 특정 base feature에 따라 변환된 feature들을 함께 볼 수 있도록...
- 시각화 : null과 zero 값은 제외하고 분포 확인, 이상치 데이터 생성 제거. -> 기존 데이터에서 군집을 나눌 수 있도록

update 2019-03-21
- 사용자가 설정한 threshold를 기준으로 데이터 분포 시각화
- 도메인 지식에 의한 추가 feature 생성 함수 추가
- whitelist는 제외하고 데이터 불러오기 추가
- 0, null 제외하고 시각화, y축 스케일 수정

update 2019-03-25
- 시각화 : null 값과 0 값의 개수를 명시!

update 2019-03-27
- sam.py 파일 완성

update 2019-03-28
- search_anomaly 함수 추가

update 2019-03-29
- 유저가 transform 함수 직접 추가 가능
- feature 엔지니어링과 상관관계 단계 코드 분리
- 전체적인 코드수정(효율적으로)

update 2019-04-01
- 상관관계 분석시 의미없는 숫자 리스트 추가(['logid','logdetailid','itemid'])

update 2019-04-02
- samtools로 클래스 외 함수 추가 : 플레이어당 feature 생성할 수 있도록 설계(ex. medal_1_level, medal_2_level,...-> medal_n_level 로 생성 가능)
- generate_agg_feature : agg_fuction을 사용자가 설정할 수 있도록 개선(goods, stats 별)
- search_anomaly -> multi-index에 따른 index_level 설정 추가

update 2019-04-03
- add_merge_base 함수 수정
- sam 객체에 local에서 수정된 df와 base을 새로 덮어쓰는 함수 추가
- sklearn robust scaler 이용하여 스케일된 값도 볼 수 있도록 추가

update 2019-04-04
- duplicate 된 column은 지우고 새로운 new_data 생성하도록 구성

update 2019-04-05
- divide_dic에서 중복된 리스트 제거 코드 추가

update 2019-04-08
- base_feature와 상관 feature간 scatter 그래프 코드 추가

update 2019-04-15
- search anomaly sort 기능 추가

update 2019-05-22
- TraitError 예외 구문 추가 -> show_new_feature_plot_scale, show_new_feature_plot
- fig_size 정하는 것 추가

update 2019-06-14
- TraitError 예외 구문 추가 -> show_joint_plot
- generate_correlation_dic 에 일정 corr 넘으면 자동선택 후 생성하는 기능 추가
################################################################################################################### by.BDH #######
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_profiling
import sklearn.preprocessing as preproc
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from scipy import stats
from random import *
import re
from itertools import chain
from collections import Counter, defaultdict
import ipywidgets as widgets
from IPython.display import display
from ipywidgets import *


## 전처리가 완료된 dataframe이 들어온다는 가정, base도 미리 설정
class Sam:
    def __init__(self,df, base):
        self.base = base
        self.df = df
        print('유지보수는 방대환님에게 문의_(bdh@netmarble.com)')

    def data_profile(self, df = None):
        if df is None:
            df = self.df
        profile = pandas_profiling.ProfileReport(df, check_correlation = False)
        self.profile = profile
        return self.profile

    def show_box_plot(self, figsize = (18,3), df = None ):
        if df is None:
            df = self.df
        self.box_figsize = figsize
        interact(self.box_plot,feature = list(df))

    def box_plot(self,feature):
        df = self.df
        figsize = self.box_figsize
        plt.figure(num=None, figsize=figsize)
        sns.set(style="whitegrid")
        ddf = df[feature].dropna()
        ddf = ddf[ddf != 0]
        static_df = self._generate_static_df(df)
        count_view = static_df[[feature,str(feature)+'_ratio']]
        print('==================================================================================')
        print(count_view)
        print('==================================================================================')
        ax = sns.boxplot(x = list(ddf.values))

    # domain 지식에 의존하여 base feature 중 합치거나, 선택하여 새로운 base feature 생성
    # ex) 불, 물, 땅, 하늘의 정수를 묶어서 하나의 정수로 생성
    def add_merge_base(self,mer_base_name, mer_base_list,df = None, base = None, is_remove_marterial = True, function = 'sum'):
        if df is None:
            df = self.df
        if base is None:
            base = self.base
        df[str(mer_base_name)] = df[mer_base_list].apply(eval(function),axis = 1)
        base.append(str(mer_base_name))
        if is_remove_marterial:
            base = list(filter(lambda x: x not in mer_base_list, base)) # 합친 base를 만들었던 재료는 제거
        self.df = df
        self.base = base
        return self.df, self.base

    ## interaction feture 생성
    def interaction_feature(self, df = None, base = None, inter_feat_lst = ['playtime']):
        if df is None:
            df = self.df
        if base is None:
            base = self.base
        for i in base:
            for j in inter_feat_lst:
                df['('+str(i)+'*'+ str(j)+')'] = df[i] * df[j]
        self.df = df
        return self.df

    ### divide_dic에 딕셔너리 추가하기
    def add_dic(self,  add_dic, divide_dic = None):
        if divide_dic is None:
            divide_dic = self.divide_dic
        dict1 = divide_dic
        dict2 = add_dic
        dict3 = defaultdict(list)
        for k, v in chain(dict1.items(), dict2.items()):
            dict3[k].extend(v)
            divide_dic = dict3
        for i,v in divide_dic.items():
            divide_dic[i] = list(set(v))
        self.divide_dic = divide_dic
        return self.divide_dic

    ## correlation 쌍 만들기
    def generate_correlation_dic(self, df= None, base = None, corr_base_default = None, corr = 0.5,auto = False):
        # 상관관계 table 형성
        if df is None:
            df = self.df
        if base is None:
            base = self.base
        try :
            df2 = df.drop(columns = ['logid','logdetailid','itemid'])
        except : df2 = df.copy()
        df_corr = df2.corr()
        divide_dic = dict()
        if not auto:
            for i_ind, i in enumerate(base):
                print('======','[',i,']','와 상관관계가 높은 list','======','\n')
                ind = df_corr[i].sort_values(ascending=[False]).head(50).index
                val = df_corr[i].sort_values(ascending=[False]).head(50).values
                da = []
                regl = str(i)+'|.+'+str(i)
                r = re.compile(regl)
                for j,v in zip(ind,val):         ## 자신은 제외
                    if str(j) != str(i) and str(i) not in list(r.findall(j)) :
                        da.append([j,round(v,4)])
                    else: pass
                temp_corr = pd.DataFrame(da,columns=['feature','correlation'])
                if len(temp_corr[temp_corr['correlation'] > corr]) == 0:
                    print('상관관계가 {}보다 큰 리스트가 없어 다음으로 이동\n'.format(corr))
                    if corr_base_default is None:           ## 상관feature default 값 입력
                        corr_base = []
                    else:
                        corr_base = corr_base_default.copy()
                    divide_dic[i] = corr_base
                    continue
                print(temp_corr[temp_corr['correlation'] > corr])
                corr_base_num = list(input('추가변수를 선택하시오.(없으면 엔터, number는","로 구분) : ').split(','))
                if corr_base_default is None:           ## 상관feature default 값 입력
                    corr_base = []
                else:
                    corr_base = corr_base_default.copy()
                if corr_base_num != ['']:
                    for k in corr_base_num:
                        corr_base.append(da[int(k)][0])
                divide_dic[i] = corr_base                                 # divide_dic 은 분석가가 선택한 base별 상관관계feature 사전
            print('===== divide_dic 생성 =====')
        else:
            for i in base:
                temp_list = list(df_corr[df_corr[i]>corr][i].keys())
                temp_list.remove(i)
                divide_dic[i] = temp_list
        self.divide_dic = divide_dic
        return self.divide_dic

    def show_joint_plot(self, divide_dic = None,df = None):
        try:
            if divide_dic is None:
                divide_dic = self.divide_dic
            if df is None:
                df = self.df
            feature_1 = Dropdown(options = list(divide_dic.keys()))
            feature_2 = Dropdown()
            def update(*args):
                feature_2.options = divide_dic[feature_1.value]
            feature_2.observe(update)
            def joint_plot(feature_1 = feature_1, feature_2 = feature_2):
                sns.set(style="white", color_codes = True)
                g = sns.jointplot(x = feature_1, y= feature_2, data= df, color ='b',size = 7)
                g.annotate(stats.pearsonr)
            interact(joint_plot,feature_1 = feature_1, feature_2 = feature_2)
        except IndexError:
            pass
        except TraitError:
            pass
        
    # 변수생성(base/상관base) 그리고 생성된 변수에 log, sqrt transfrom feature 생성하는 함수
    def _ft_trans(self, df, base, corr_base, function_lst):
        if len([base]) == 1:
            base = [base]
        df1 = df[base]
        corr_base = list(corr_base)
        if corr_base != []:
            ### Feature transform 함수 추가 가능 ###
            # 1단계
            for i in base:
                for j in corr_base:
                    if i != j:
                        df1[i + '/' + str(j)] = df1[i] / (df[j] + 1)
        self.notfunc_lst = list(df1)
        # 2단계
        ## 원하는 함수 추가 가능
        for k in list(df1):
            if len(function_lst) == 0:
                break
            for fun in function_lst:
                df1[str(fun) + '(' + k + ')'] = df1[k].apply(lambda x: eval(fun)(x))
        return df1

    # 상관관계를 보기 전
    # 상관관계는 base 증가 -> feature*playtime 이 유의미한 경우 발생!
    ## feature engineering
    def feature_engineering(self, df= None, base = None, divide_dic = None, function = None, isfunctionAdd = None):
        # 상관관계 table 형성
        if df is None:
            df = self.df
        if base is None:
            base = self.base
        if divide_dic is None:
            divide_dic = self.divide_dic
        if isfunctionAdd is None:
            isfunctionAdd = False
        if function is None:
            function_lst = ['np.log1p','np.sqrt']
        elif isfunctionAdd == True:
            function_lst = ['np.log1p','np.sqrt']
            function_lst.extend(function)
        else:
            function_lst = list(function)
        for i_ind, i in enumerate(base):
            if len(divide_dic[i]) == 0 :
                corr_base = []
            else:
                corr_base = divide_dic[i]               ## 상관feature default 값 입력

            transf_data = self._ft_trans(df,i,corr_base,function_lst)
            if i_ind == 0 :
                new_data = transf_data
            else:
                new_data = pd.concat([new_data, transf_data],axis = 1)
            divide_dic[i] = corr_base                                  # divide_dic 은 분석가가 선택한 base별 상관관계feature 사전
        print('===== new_data 생성 =====')
        new_data = new_data.loc[:,~new_data.columns.duplicated()]        # 혹시 모르게 duplicat가 생긴 column을 지워주는 역할
        self.df = new_data
        return self.df


    ## optional attribute columns 추출 및 feature engineering
    ###########################################################################################
    ###                            2단계, 3단계 동시 수행                                   ###
    ###      상관관계를 이용해 feature를 추출하고 feature engineering까지 완료하는 함수     ###
    ###########################################################################################
    def feature_engineering_with_corr(self, df= None, base = None, corr_base_default = None, corr = None, function = None, isfunctionAdd = None):
        # 상관관계 table 형성
        if df is None:
                df = self.df
        if base is None:
            base = self.base
        if isfunctionAdd is None:
            isfunctionAdd = False
        if function is None:
            function_lst = ['np.log1p','np.sqrt']
        elif isfunctionAdd == True:
            function_lst = ['np.log1p','np.sqrt']
            function_lst.extend(function)
        else:
            function_lst = list(function)
        divide_dic = self.generate_correlation_dic(df, base, corr_base_default,corr)

        for i_ind, i in enumerate(base):
            if len(divide_dic[i]) == 0 :
                corr_base = []
            else:
                corr_base = divide_dic[i]

            transf_data = self._ft_trans(df,i,corr_base,function_lst)
            if i_ind == 0 :
                new_data = transf_data
            else:
                new_data = pd.concat([new_data, transf_data],axis = 1)
        new_data = new_data.loc[:,~new_data.columns.duplicated()]        # 혹시 모르게 duplicat가 생긴 column을 지워주는 역할
        self.df = new_data
        self.divide_dic = divide_dic
        print('===== new_data 생성 =====')
        return self.df, self.divide_dic

    #### null, zero count 를 기록해 놓는 코드!
    def _generate_static_df(self,data):
        for indx,feature in enumerate(list(data)):
            graph = data[feature].dropna().copy()
            tota = data[feature].shape[0]
            count_view = pd.DataFrame({'Total data' : [data[feature].shape[0],round(data[feature].shape[0]/tota, 2)],
                                       'Null' : [data[feature].isnull().sum(), round(data[feature].isnull().sum()/tota,2)],
                                       'Zero': [data[data[feature] == 0].count(axis = 0)[feature], round(data[data[feature] == 0].count(axis = 0)[feature]/tota, 2)],
                                       'Expressed Data' :[len(graph[graph != 0].values),round(len(graph[graph != 0].values)/tota,2)]}, index = [str(feature),str(feature)+ '_ratio']).T
            if indx == 0 :
                static_df = count_view
            else:
                static_df = pd.concat([static_df, count_view], axis = 1)
        return static_df

    def show_visualize_data(self,df = None, thres = None):
        if df is None:
            df= self.df
        if thres is None:
            thres = range(0,100000)
        interact(self.visualize_data,feature = list(df),thres = thres)

    def visualize_data(self,feature,thres):
        data = self.df
        fig = plt.figure()
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 2))
        graph = data[feature].dropna().copy()
        tota = data[feature].shape[0]
        static_df = self._generate_static_df(data)
        count_view = static_df[[feature,str(feature)+'_ratio']]
        print('===================================================================================================================')
        print(count_view)
        print('===================================================================================================================')
        print('Min :',round(data[feature].min(),4),'\t','\t','Median : ',round(data[feature].median(),4),'\t','\t','Max :', round(data[feature].max(),4),'\t','\t','Mean :', round(data[feature].mean(),4))
        print('===================================================================================================================')
        sns.set(style="whitegrid")
        ax1 = sns.boxplot(x = list(graph[graph != 0].values))
        ax1.set_title('<'+str(feature)+'  '+'boxplot'+'>', fontsize=17)
        ## ===== 다른 plot ===== ##
        new_data_1 = graph[graph <= thres]
        new_data_2 = graph[graph > thres]
        figg = plt.figure()
        figg, ax2 = plt.subplots(1, 1, figsize=(9, 6))
        bins = 30
        ax2.hist(new_data_1[new_data_1 != 0], bins= bins,lw=2, ec="green", fc=(0, 0, 0, 0.3), label ='data')
        ax2.hist(new_data_2[new_data_2 != 0], bins= bins,lw=2, ec="blue", fc=(0, 0, 1, 0.3), label ='data_outlier')
        ax2.tick_params(labelsize=14)
        ax2.set_xlabel(str(feature), fontsize=14)
        ax2.set_ylabel('frequency', fontsize=14)
        ax2.set_title('<'+str(feature)+'  '+'histogram'+'>', fontsize=17)
        ax2.legend()

    def show_compare_plot(self,df = None, base = None):
        if df is None:
            df= self.df
        if base is None:
            base = self.base
        interact(self.compare_plot,feature = list(base))

    def compare_plot(self, feature):
        df = self.df
        base = self.base
        dict_f = self._similar_name_dic(df, base)
        cnt = 0
        cnt_z = 0
        bins = 30
        if len(dict_f[feature]) % 2 == 0:
            k = len(dict_f[feature]) // 2
        else:
            k = len(dict_f[feature]) // 2 + 1
        fig, ax = plt.subplots(k, 2,figsize=(20,7*k))
        for i_ind,i in enumerate(dict_f[feature]):
            graph = df[i].dropna().copy()
            ax[cnt,cnt_z].hist(graph[graph != 0], bins= bins,lw=2, ec="blue", fc=(0, 0, 1, 0.3), label ='data')
            ax[cnt,cnt_z].tick_params(labelsize=14)
            ax[cnt,cnt_z].set_xlabel(str(i), fontsize=14)
            ax[cnt,cnt_z].set_ylabel('frequency', fontsize=14)
            ax[cnt,cnt_z].legend()
            if cnt_z == 0 :
                cnt_z = 1
            else:
                cnt += 1
                cnt_z = 0
            if i_ind == len(dict_f[feature]) - 1 : break

    ### 생성된 new_data feature를 base에 대하여 매칭한 dictionary ###
    def _similar_name_dic(self, df, base):
        dict_f = {}
        for i in base:
            regl = i+'|.+'+i
            r = re.compile(regl)
            dict_f[i] = list(filter(r.match,list(df)))
        return dict_f

    
    ####### agg_feature 객체 따로 생성 안할려면...
    #######  other를 이용하면 되긴함

    
    def generate_agg_feature(self, df = None, function = None, base = None):
        if df is None :
            df = self.df
        if base is None:
            base = self.base
        if function is None:
            function = ['sum','max','median','min','mean', 'std','skew','count']
        else:
            function = list(function)
        if df.index.name != 'pid' :
            df = df.reset_index()
            df = df.set_index('pid')
        agg_df = df.groupby('pid')[base].agg(function)
        agg_df_col = []
        for i in list(agg_df):
            agg_df_col.append(i[1]+'_('+i[0]+')')
        agg_df.columns = agg_df_col
        self.df = agg_df
        return self.df

    def feature_selection(self, df = None):
        if df is None :
            df = self.df
        final_feature = []
        for v in (list(df)):
            print('========',v,'========')
            a = input('선택:1, 선택안함:엔터, 끝내기:q' + '\n').split()
            if len(a) != 0 and a[0][0] == '1':                         #11을 눌러도 등록이 될 수 있게
                final_feature.append(v)
            if len(a) != 0 and a[0][0] == 'q':
                      break
            print('selected Feature : ', final_feature)
            print()
        print('=== Final_data ===')
        final_data_sample = df[final_feature]
        self.final_data_sample = final_data_sample
        self.final_feature = final_feature
        return self.final_feature, self.final_data_sample

    def generate_data(self, total_df, final_feature = None):
        if final_feature is None:
            final_feature = self.final_feature
        df = total_df
        p = re.compile('\W+|\w+')
        features = list(df)
        for i in final_feature:
            m1 = p.findall(i)
            string = []
            barr_num = 0
            func = ''
            for ind_k, k in enumerate(m1):
                if k == '(':
                    func = ''.join(m1[:ind_k])
                    barr_num = ind_k
                if k in features:
                    string.append("df"+"['"+ str(k) +"']")
                else:
                    string.append(k)
            string = string[barr_num:]
            string = ''.join(string)
            if func != '':
                df[i] = eval(string).apply(lambda x: eval(func)(x))
            else:
                df[i] = eval(string)
        return df[final_feature]

    ### 이상치 데이터(n개) 생성 함수
    # df: 생성 원천 소스, base : base변수, divide_dic : base변수의 상관변수리스트, n : 이상치 샘플 수, weight_a,weight_b: 이상치 생성 범위(배수)
    def generate_anomaly(self, df, base, divide_dic, n, weight_range = (1,2)):
        for i_index,i in enumerate(base):
            for num in range(n):
                if i in list(divide_dic):
                    corr_data = pd.Series(df[divide_dic[i]].mean(axis=0))
                    k = uniform(weight_range[0],weight_range[1])                                 ## 이상치 가중치 주는 코드 default : (1,2)
                    q = uniform(0.5, 1.0)
                    new_anom_1 = pd.concat([pd.Series(df[i].max(axis=0)*k,index =[i]), corr_data*q], axis =0)
                else:
                    k = uniform(weight_range[0],weight_range[1])
                    new_anom_1 = pd.Series(df[i].max(axis=0)*k,index = [i])
                if num == 0 :
                    anomaly_data = new_anom_1
                else:
                    anomaly_data = pd.concat([anomaly_data, new_anom_1],axis = 1)
            anomaly_data = anomaly_data.T
            anomaly_data = self._ft_trans(anomaly_data,i,divide_dic[i])
            if i_index == 0 :
                total_anomaly_data = anomaly_data
            else:
                total_anomaly_data = pd.concat([total_anomaly_data,anomaly_data],axis = 1)
        self.total_anomaly_data = total_anomaly_data
        return self.total_anomaly_data

    def search_anomaly(self,feature_name, threshold, index_level = None, df = None):
        if index_level is None:
            index_level = 0
        if df is None:
            new_data = self.df
        else:
            new_data = df
        a = list(new_data[new_data[feature_name] >= threshold].index)
        d = Counter(a)
        f_df = new_data[new_data[feature_name] >= threshold][feature_name].groupby(level= index_level).sum()
        f_df_mean = new_data[new_data[feature_name] >= threshold][feature_name].groupby(level= index_level).mean()
        ff_df = pd.Series(d)
        fff_df = pd.concat([ff_df,f_df],axis = 1)
        fff_df = pd.concat([fff_df,f_df_mean],axis = 1)
        fff_df.columns = ['중복 count','sum_('+str(feature_name)+')','mean_('+str(feature_name)+')']
        fff_df = fff_df.sort_values(by=['mean_('+str(feature_name)+')','sum_('+str(feature_name)+')'],ascending = False)
        return fff_df

    def scale_data(self, df = None, scaler = None):
        if df is None:
            df = self.df
        if scaler is None:
            scaler = RobustScaler()
        else:
            scaler = eval(scaler)
        s_data = scaler.fit_transform(df)
        df_scale = pd.DataFrame(s_data,columns = list(df),index = df.index)
        self.scale_df = df_scale
        return self.scale_df

    def show_visualize_data_scale(self,data = None, thres = None):
        if data is None:
            data = self.scale_df
        if thres is None:
            thres = range(-20,20)
        interact(self.visualize_data_scale,feature = list(data),thres = thres)

    def show_compare_plot_scale(self,data = None, base = None):
        if data is None:
            data = self.scale_df
        if base is None:
            base = self.base
        interact(self.compare_plot_scale,feature = list(base))

    def visualize_data_scale(self,feature,thres):
        data = self.scale_df
        fig = plt.figure()
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 2))
        graph = data[feature].dropna().copy()
        tota = data[feature].shape[0]
        static_df = self._generate_static_df(data)
        count_view = static_df[[feature,str(feature)+'_ratio']]
        print('===================================================================================================================')
        print(count_view)
        print('===================================================================================================================')
        print('Min :',round(data[feature].min(),4),'\t','\t','Median : ',round(data[feature].median(),4),'\t','\t','Max :', round(data[feature].max(),4),'\t','\t','Mean :', round(data[feature].mean(),4))
        print('===================================================================================================================')
        sns.set(style="whitegrid")
        ax1 = sns.boxplot(x = list(graph.values))
        ax1.set_title('<'+'scale_'+str(feature)+'  '+'boxplot'+'>', fontsize=17)
        ## ===== 다른 plot ===== ##
        new_data_1 = graph[graph <= thres]
        new_data_2 = graph[graph > thres]
        figg = plt.figure()
        figg, ax2 = plt.subplots(1, 1, figsize=(9, 6))
        bins = 30
        ax2.hist(new_data_1, bins= bins,lw=2, ec="green", fc=(0, 0, 0, 0.3), label ='data')
        ax2.hist(new_data_2, bins= bins,lw=2, ec="blue", fc=(0, 0, 1, 0.3), label ='data_outlier')
        ax2.tick_params(labelsize=14)
        ax2.set_xlabel(str(feature), fontsize=14)
        ax2.set_ylabel('frequency', fontsize=14)
        ax2.set_title('<'+'scale_'+str(feature)+'  '+'histogram'+'>', fontsize=17)
        ax2.legend()

    def compare_plot_scale(self, feature):
        df = self.scale_df
        base = self.base
        dict_f = self._similar_name_dic(df, base)
        bins = 30
        if len(dict_f[feature]) % 2 == 0:
            k = len(dict_f[feature]) // 2
        else:
            k = len(dict_f[feature]) // 2 + 1
        fig, ax = plt.subplots(k, 2,figsize=(20,7*k))
        cnt = 0
        cnt_z = 0
        for i_ind,i in enumerate(dict_f[feature]):
            graph = df[i].copy()
            ax[cnt,cnt_z].hist(graph, bins= bins,lw=2, ec="blue", fc=(0, 0, 1, 0.3), label ='scale_data')
            ax[cnt,cnt_z].tick_params(labelsize=14)
            ax[cnt,cnt_z].set_xlabel(str(i), fontsize=14)
            ax[cnt,cnt_z].set_ylabel('frequency', fontsize=14)
            ax[cnt,cnt_z].legend()
            if cnt_z == 0 :
                cnt_z = 1
            else:
                cnt += 1
                cnt_z = 0
            if i_ind == len(dict_f[feature]) - 1 : break

    def show_new_feature_plot(self, df = None, base = None, threshold = None, bar_figsize = None, plot_figsize = None):
        try :
            if df is None:
                df = self.df
            if threshold is None:
                threshold = range(0,10000)
            if base is None:
                base = self.base
            if bar_figsize is None:
                bar_figsize = (12,2)
            if plot_figsize is None:
                plot_figsize = (9,6)
    #         base = self.notfunc_lst               # transform 함수 적용으로 분류해서 보려면 이것 사용
            feature_1 = Dropdown(options = base)
            feature_2 = Dropdown()
            def update(*args):
                dict_f = self._similar_name_dic(df, base)
                feature_2.options = dict_f[feature_1.value]
            feature_2.observe(update)
            def visualize(base = feature_1,feature = feature_2, threshold = threshold):
                data = df
                fig = plt.figure()
                fig, ax1 = plt.subplots(1, 1, figsize=bar_figsize)
                graph = data[feature].dropna().copy()
                tota = data[feature].shape[0]
                static_df = self._generate_static_df(data)
                count_view = static_df[[feature,str(feature)+'_ratio']]
                print('===================================================================================================================')
                print(count_view)
                print('===================================================================================================================')
                print('Min :',round(data[feature].min(),4),'\t','\t','Median : ',round(data[feature].median(),4),'\t','\t','Max :', round(data[feature].max(),4),'\t','\t','Mean :', round(data[feature].mean(),4))
                print('===================================================================================================================')
                sns.set(style="whitegrid")
                ax1 = sns.boxplot(x = list(graph[graph != 0].values))
                ax1.set_title('<'+str(feature)+'  '+'boxplot'+'>', fontsize=17)
                ## ===== 다른 plot ===== ##
                new_data_1 = graph[graph <= threshold]
                new_data_2 = graph[graph > threshold]
                figg = plt.figure()
                figg, ax2 = plt.subplots(1, 1, figsize=plot_figsize)
                bins = 30
                ax2.hist(new_data_1[new_data_1 != 0], bins= bins,lw=2, ec="green", fc=(0, 0, 0, 0.3), label ='data')
                ax2.hist(new_data_2[new_data_2 != 0], bins= bins,lw=2, ec="blue", fc=(0, 0, 1, 0.3), label ='data_outlier')
                ax2.tick_params(labelsize=14)
                ax2.set_xlabel(str(feature), fontsize=14)
                ax2.set_ylabel('frequency', fontsize=14)
                ax2.set_title('<'+str(feature)+'  '+'histogram'+'>', fontsize=17)
                ax2.legend()
            interact(visualize, base = feature_1, feature = feature_2, threshold = threshold)
        except TraitError:
            pass
        
    def get_df(self):
        return self.df

    def get_base(self):
        return self.base

    def get_profile(self):
        return self.profile

    def get_final_feature(self):
        return self.final_feature

    ### 금지 함수 ###
    def input_df(self, df):
        self.df = df
    def input_base(self, base):
        self.base = base
    
###########################################################################
#############################
### samtools ### -> no class
############################
def generate_multi_feature_stat(n,feature_name_list):
    new_feature = []
    for i in range(1,n+1):
        for j in feature_name_list:
            new_feature.append(j.replace('n',str(i)))
    return new_feature

def generate_multi_feature_goods(feature_name_list,general = None):
    new_feature = []
    if general is None:
        general = ['my','get','use']
    else : general = list(general)
    for i in feature_name_list:
        for j in general:
            new_feature.append(j+i)
    return new_feature

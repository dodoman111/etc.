def parallelPreprocess(DATASET) {
    parallel (
        'mode1' : {
            MODE = '1'
            sh """./3_new_preprocess_ext.sh ${PROJECT_NAME} ${GAMECODE} ${MODEL_TYPE} ${JOB_DATE} ${MODE} ${DATASET} """
        },
        'mode2' : {
            sleep 10
            MODE = '2'
            sh """./3_new_preprocess_ext.sh ${PROJECT_NAME} ${GAMECODE} ${MODEL_TYPE} ${JOB_DATE} ${MODE} ${DATASET}"""
        },
        'mode3' : {
            sleep 20
            MODE = '3'
            sh """./3_new_preprocess_ext.sh ${PROJECT_NAME} ${GAMECODE} ${MODEL_TYPE} ${JOB_DATE} ${MODE} ${DATASET}"""
        },
        'mode4' : {
            sleep 30
            MODE = '4'
            sh """./3_new_preprocess_ext.sh ${PROJECT_NAME} ${GAMECODE} ${MODEL_TYPE} ${JOB_DATE} ${MODE} ${DATASET}"""
        },
        'mode5' : {
            sleep 40
            MODE = '5'
            sh """./3_new_preprocess_ext.sh ${PROJECT_NAME} ${GAMECODE} ${MODEL_TYPE} ${JOB_DATE} ${MODE} ${DATASET}"""
        }
    )
}
/*
//hyper parameter tuning
def parallelTrain(DATASET) {
    parallel (
        'mode1' : {
            MODE = '1'
            sh """./4-0_hptune_training.sh ${PROJECT_NAME} ${GAMECODE} ${MODEL_TYPE} ${JOB_DATE} ${MODEL_DATETIME_TRAIN} ${DATASET} ${MODE} """
        },
        'mode2' : {
            sleep 10
            MODE = '2'
            sh """./4-0_hptune_training.sh ${PROJECT_NAME} ${GAMECODE} ${MODEL_TYPE} ${JOB_DATE} ${MODEL_DATETIME_TRAIN} ${DATASET} ${MODE} """
        },
        'mode3' : {
            sleep 20
            MODE = '3'
            sh """./4-0_hptune_training.sh ${PROJECT_NAME} ${GAMECODE} ${MODEL_TYPE} ${JOB_DATE} ${MODEL_DATETIME_TRAIN} ${DATASET} ${MODE} """
        },
        'mode4' : {
            sleep 30
            MODE = '4'
            sh """./4-0_hptune_training.sh ${PROJECT_NAME} ${GAMECODE} ${MODEL_TYPE} ${JOB_DATE} ${MODEL_DATETIME_TRAIN} ${DATASET} ${MODE} """
        },
        'mode5' : {
            sleep 40
            MODE = '5'
            sh """./4-0_hptune_training.sh ${PROJECT_NAME} ${GAMECODE} ${MODEL_TYPE} ${JOB_DATE} ${MODEL_DATETIME_TRAIN} ${DATASET} ${MODE} """
        }
    )
}
*/

def parallelTrain(DATASET) {
    parallel (
        'mode1' : {
            MODE = '1'
            sh """./4-1_training.sh ${PROJECT_NAME} ${GAMECODE} ${MODEL_TYPE} ${JOB_DATE} ${MODEL_DATETIME_TRAIN} ${DATASET} ${MODE} """
        },
        'mode2' : {
            sleep 10
            MODE = '2'
            sh """./4-2_training.sh ${PROJECT_NAME} ${GAMECODE} ${MODEL_TYPE} ${JOB_DATE} ${MODEL_DATETIME_TRAIN} ${DATASET} ${MODE} """
        },
        'mode3' : {
            sleep 20
            MODE = '3'
            sh """./4-3_training.sh ${PROJECT_NAME} ${GAMECODE} ${MODEL_TYPE} ${JOB_DATE} ${MODEL_DATETIME_TRAIN} ${DATASET} ${MODE} """
        },
        'mode4' : {
            sleep 30
            MODE = '4'
            sh """./4-4_training.sh ${PROJECT_NAME} ${GAMECODE} ${MODEL_TYPE} ${JOB_DATE} ${MODEL_DATETIME_TRAIN} ${DATASET} ${MODE} """
        },
        'mode5' : {
            sleep 40
            MODE = '5'
            sh """./4-5_training.sh ${PROJECT_NAME} ${GAMECODE} ${MODEL_TYPE} ${JOB_DATE} ${MODEL_DATETIME_TRAIN} ${DATASET} ${MODE} """
        }
    )
}

def parallelPredict(DATASET) {
    parallel(
        'mode1' : {
            MODE = '1'
            sh """./5_predict.sh ${PROJECT_NAME} ${GAMECODE} ${MODEL_TYPE} ${JOB_DATE} ${MODEL_DATETIME_TRAIN} ${MODEL_DATETIME_PRED} ${DATASET} ${MODE}"""
        },
        'mode2' : {
            sleep 10
            MODE = '2'
            sh """./5_predict.sh ${PROJECT_NAME} ${GAMECODE} ${MODEL_TYPE} ${JOB_DATE} ${MODEL_DATETIME_TRAIN} ${MODEL_DATETIME_PRED} ${DATASET} ${MODE}"""
        },
        'mode3' : {
            sleep 20
            MODE = '3'
            sh """./5_predict.sh ${PROJECT_NAME} ${GAMECODE} ${MODEL_TYPE} ${JOB_DATE} ${MODEL_DATETIME_TRAIN} ${MODEL_DATETIME_PRED} ${DATASET} ${MODE}"""
        },
        'mode4' : {
            sleep 30
            MODE = '4'
            sh """./5_predict.sh ${PROJECT_NAME} ${GAMECODE} ${MODEL_TYPE} ${JOB_DATE} ${MODEL_DATETIME_TRAIN} ${MODEL_DATETIME_PRED} ${DATASET} ${MODE}"""
        },
        'mode5' : {
            sleep 40
            MODE = '5'
            sh """./5_predict.sh ${PROJECT_NAME} ${GAMECODE} ${MODEL_TYPE} ${JOB_DATE} ${MODEL_DATETIME_TRAIN} ${MODEL_DATETIME_PRED} ${DATASET} ${MODE}"""
        }
    )
}

def parallelBqload(DATASET) {
    parallel (
        'mode1' : {
            MODE='1'
            sh """./6_bq_load.sh ${PROJECT_NAME} ${GAMECODE} ${MODEL_TYPE} ${JOB_DATE} ${MODEL_DATETIME_PRED} ${DATASET} ${MODE}"""
        },
        'mode2' : {
            sleep 15
            MODE='2'
            sh """./6_bq_load.sh ${PROJECT_NAME} ${GAMECODE} ${MODEL_TYPE} ${JOB_DATE} ${MODEL_DATETIME_PRED} ${DATASET} ${MODE}"""
        },
        'mode3' : {
            sleep 30
            MODE='3'
            sh """./6_bq_load.sh ${PROJECT_NAME} ${GAMECODE} ${MODEL_TYPE} ${JOB_DATE} ${MODEL_DATETIME_PRED} ${DATASET} ${MODE}"""
        },
        'mode4' : {
            sleep 45
            MODE='4'
            sh """./6_bq_load.sh ${PROJECT_NAME} ${GAMECODE} ${MODEL_TYPE} ${JOB_DATE} ${MODEL_DATETIME_PRED} ${DATASET} ${MODE}"""
        },
        'mode5' : {
            sleep 60
            MODE='5'
            sh """./6_bq_load.sh ${PROJECT_NAME} ${GAMECODE} ${MODEL_TYPE} ${JOB_DATE} ${MODEL_DATETIME_PRED} ${DATASET} ${MODE}"""
        }
    )
}

node {
    try{
        // JOB_DATE = '20190805'
        JOB_DATE = sh(script: 'TZ=Asia/Seoul date "+%Y%m%d" -d "yesterday"', returnStdout: true).trim()
        BLOCKED_CHECK_DATE = sh(script: 'TZ=Asia/Seoul date "+%Y%m%d" -d "7 days ago"', returnStdout: true).trim()
        timeout(time:15, unit:'MINUTES') {
            stage('Initialize') {
                echo "MODE:             ${MODE}"
                echo "MODEL_TYPE:       ${MODEL_TYPE}"
                echo "PROJECT_NAME:     ${PROJECT_NAME}"
                echo "JOB_DATE:         ${JOB_DATE}"
                echo "REPORT_LIMIT_NUM: ${REPORT_LIMIT_NUM}"
            }
            stage('Code_Sync') {
                retry(3) {
                  sh """
                        if [ ! -d ".git" ]; then
                            rm -rf ./*
                            git config --global user.name "daehwanbang"
                            git config --global user.email "bdh@netmarble.com"
                            git clone git@10.128.0.4:di/anomaly-detection-${MODE}-${MODEL_TYPE} ./
                            CURRENT_PATH=`pwd`
                            chmod +x \$CURRENT_PATH/script/*.sh
                        else
                            git reset --hard origin
                            git pull origin master
                            CURRENT_PATH=`pwd`
                            chmod +x \$CURRENT_PATH/script/*.sh
                        fi
                   """
                }
            }
        }
        timeout(time:2, unit:'HOURS') {
            dir('script') {
                stage('Preprocess - Generate stats feature data') {
                    retry(3) {
                        sh """./2_preprocess_gen.sh ${PROJECT_NAME} ${MODE} ${MODEL_TYPE} ${JOB_DATE}"""
                    }
                }
                stage('Preprocess - Extract train/test/predict data') {
                    retry(3) {
                        parallel (
                            'k1' : {
                                DATASET='k1'
                                parallelPreprocess(DATASET)
                            },
                            'k2' : {
                                sleep 10
                                DATASET='k2'
                                parallelPreprocess(DATASET)
                            },
                            'k3' : {
                                sleep 20
                                DATASET='k3'
                                parallelPreprocess(DATASET)
                            }
                        )
                    }
                }
            }
        }
        timeout(time:2, unit:'HOURS') {
            dir('script') {
                stage('Train') {
                    retry(3) {
                        MODEL_DATETIME_TRAIN = sh(script: 'date "+%s"', returnStdout: true).trim()
                        echo 'training...'
                        parallel (
                            'k1' : {
                                DATASET='k1'
                                parallelTrain(DATASET)
                            },
                            'k2' : {
                                sleep 10
                                DATASET='k2'
                                parallelTrain(DATASET)
                            },
                            'k3' : {
                                sleep 20
                                DATASET='k3'
                                parallelTrain(DATASET)
                            }
                        )
                    }
                }
                stage('Predict') {
                    retry(3) {
                        MODEL_DATETIME_PRED = sh(script: 'date "+%s"', returnStdout: true).trim()
                        echo 'predicting...'
                        parallel (
                            'k1' : {
                                DATASET='k1'
                                parallelPredict(DATASET)
                            },
                            'k2' : {
                                sleep 10
                                DATASET='k2'
                                parallelPredict(DATASET)
                            },
                            'k3' : {
                                sleep 20
                                DATASET='k3'
                                parallelPredict(DATASET)
                            }
                        )
                    }
                }
            }
        }
        timeout(time:50, unit:'MINUTES') {
            dir('script') {
                stage('Bq load') {
                    retry(3) {
                        echo 'initialize predict result'
                        sh """
                            bq rm --project_id=${PROJECT_NAME} -f -t ${MODE}_${MODEL_TYPE}.${MODEL_TYPE}_predict_result_${JOB_DATE}
                        """
                        echo 'Done : initialize predict result'
                        sleep 60
                        echo 'BQ load'
                        parallel (
                            'k1' : {
                                DATASET='k1'
                                parallelBqload(DATASET)
                            },
                            'k2' : {
                                sleep 15
                                DATASET='k2'
                                parallelBqload(DATASET)
                            },
                            'k3' : {
                                sleep 30
                                DATASET='k3'
                                parallelBqload(DATASET)
                            }
                        )
                    }
                }
            }
        }
        timeout(time:50, unit:'MINUTES') {
            dir('report') {
                stage('report') {
                    sh """
                        python3 report.py --job_date ${JOB_DATE} --report_limit_num ${REPORT_LIMIT_NUM} --threshold ${THRESHOLD}
                    """
                }
            }
            stage('post') {
                always {
                    echo 'Job finished...'
                }
            }
        }
        boolean nextjob = false
        stage('BQ load : detection results to di-report') {
            try{
                sh """
                    bq query \
                       --project_id=nm-dataintelligence \
                       --use_legacy_sql=false \
                       --replace=false \
                       --max_rows=3 \
                       --destination_table=di_report.detection_report_${JOB_DATE} \
                       "SELECT '${JOB_DATE}' as job_date, '${MODE}' as game_code, '${MODEL_TYPE}' as model_type, IFNULL(COUNT(1),0) as detection_cnt, 0 as blocked_cnt FROM (SELECT * FROM (SELECT *, ROW_NUMBER() OVER (PARTITION BY result.pid, result.pcseq ORDER BY result.score DESC) AS rn FROM \\`${PROJECT_NAME}.${GAMECODE}_${MODEL_TYPE}.${MODEL_TYPE}_predict_result_${JOB_DATE}\\` result ORDER BY score DESC LIMIT ${REPORT_LIMIT_NUM}) WHERE score>=${THRESHOLD} AND rn=1)"
                """
            }
            catch(error){
                nextjob = true
                currentBuild.result='SUCCESS'
            }
            if(nextjob) {
                echo "nextjob execute"
                sh """ bq query \
                       --project_id=nm-dataintelligence \
                       --use_legacy_sql=false \
                       --replace=false \
                       --max_rows=3 \
                       "INSERT di_report.detection_report_${JOB_DATE} SELECT '${JOB_DATE}' as job_date, '${MODE}' as game_code, '${MODEL_TYPE}' as model_type, IFNULL(COUNT(1),0) as detection_cnt, 0 as blocked_cnt FROM (SELECT * FROM (SELECT *, ROW_NUMBER() OVER (PARTITION BY result.pid, result.pcseq ORDER BY result.score DESC) AS rn FROM \\`${PROJECT_NAME}.${GAMECODE}_${MODEL_TYPE}.${MODEL_TYPE}_predict_result_${JOB_DATE}\\` result ORDER BY score DESC LIMIT ${REPORT_LIMIT_NUM}) WHERE score>=${THRESHOLD} AND rn=1)"
                """
            }
            sh """
                python3 ./detection/blocked_update.py --job_date=${JOB_DATE} --blocked_check_date=${BLOCKED_CHECK_DATE}
            """
        }
    }
    catch(err) {
        currentBuild.result = 'FAILURE'
    }
    finally {
        step([$class: 'Mailer', notifyEveryUnstableBuild: true, recipients: RECIPIENTS, sendToIndividuals: true])
    }
}

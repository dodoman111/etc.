with
  logs AS(
      SELECT
        CASE MOD(CAST(RAND()*10 AS INT64),3)
          WHEN  1 THEN 'k1' 
          WHEN  2 THEN 'k2'
          ELSE 'k3' END AS dataset,
          * EXCEPT(rn,mode),
        CASE
          WHEN mode in (1,9,10,12,13,14,15)    THEN '1'
          WHEN mode in (3,6)                   THEN '2'
          WHEN mode in (2)                     THEN '3'
          WHEN mode in (7)                     THEN '4'
          WHEN mode in (21,22,23,24,25)        THEN '5'
          ELSE '6' END AS mode
      FROM(
       SELECT 
          *,ROW_NUMBER() OVER (PARTITION BY logkey ORDER BY regdatetime) AS rn
       FROM 
          `Log_*`
       where 
         7 = 7
         AND _TABLE_SUFFIX = @job_date
         AND logid = 3
         AND logdetailid =2
         AND result = 'S'
     )
     WHERE
        7=7
        AND rn = 1
    ), 
  _log AS (
      SELECT *
      FROM logs LEFT JOIN `whitelist` W using(pid)
      WHERE W.pid is null
    ),
  log_feature AS(
   // 관련된 로그들  
  ),
  log_feature_long as (
    SELECT 
        dataset, mode, pid, pcseq, regdatetime, 
        arr_values.name as name, arr_values.value as value
    FROM (
        SELECT
        dataset, mode, pid, pcseq, regdatetime, 
        [
            STRUCT('ft01'                        AS name, ft01                            AS value),
            STRUCT('ft02'                        AS name, ft02                            AS value),
            STRUCT('ft03'                        AS name, ft03                            AS value)] AS arr_values
        FROM stats
        ), 
    UNNEST(arr_values) AS arr_values
    ),
  log_scaler AS (
    SELECT
      mode, name,
      AVG(value)    AS value_mean,
      STDDEV(value) AS value_stddev,
      ANY_VALUE(value_limit) AS value_limit
    FROM (
        SELECT mode,name, value, PERCENTILE_CONT(value, 0.9) OVER(PARTITION BY mode, name) AS value_limit
        FROM stats_long
        )
    WHERE
      value <= value_limit
    GROUP BY mode,name
    ),
  outlier_info AS (
    SELECT DISTINCT
        log_feature_long.mode, log_feature_long.pid, log_feature_long.pcseq, 
        log_feature_long.regdatetime,
        log_feature_long.dataset,
        'k0' AS new_dataset
    FROM log_feature_long LEFT JOIN log_scaler USING(mode,name)
    WHERE log_feature_long.value > log_scaler.value_limit
    ),
  log_scaled AS (
    SELECT 
      stat.* EXCEPT(dataset), 
      CASE 
      WHEN outlier_info.new_dataset IS NOT NULL THEN outlier_info.new_dataset ELSE stat.dataset END AS dataset
    FROM( 
        SELECT
            log_feature_long.dataset,logkey,
            log_feature_long.mode, log_feature_long.pid, log_feature_long.pcseq, 
            log_feature_long.regdatetime, log_feature_long.medal_num,
            log_feature_long.name, log_feature_long.value, log_scaled.value_limit,
            CASE 
            WHEN stats_scaler.value_stddev = 0 
              THEN log_feature_long.value - log_scaled.value_mean
            ELSE (log_feature_long.value - log_scaled.value_mean)/log_scaled.value_stddev
            END AS value_scaled
        FROM stats_long 
        LEFT JOIN log_scaler USING(mode,name)
        ) stat
    LEFT JOIN outlier_info USING(dataset,mode,pid,pcseq,regdatetime,medal_num)
     )
     
    SELECT 
      dataset,
      pid, pcseq, regdatetime, mode, medal_num,
      ft01,ft02,ft03
    FROM(
      SELECT
          dataset,mode, pid, pcseq, regdatetime,
          ANY_VALUE(CASE WHEN name = 'ft01'                        THEN value_scaled END) AS ft01,
          ANY_VALUE(CASE WHEN name = 'ft02'                        THEN value_scaled END) AS ft02,
          ANY_VALUE(CASE WHEN name = 'ft03'                        THEN value_scaled END) AS ft03
      FROM log_scaled
      GROUP BY dataset,mode, pid, pcseq, regdatetime
    )

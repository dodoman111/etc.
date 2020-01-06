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
          _TABLE_SUFFIX = @job_date
     )
     WHERE
         rn = 1
    ), 
  _log AS (
      SELECT *
      FROM logs LEFT JOIN `whitelist` W using(pid)
      WHERE W.pid is null
    ),
  log_feature AS(
   // 관련된 로그들  // 전처리 // 아웃라이어 제거 // 스케일러의 과정을 거침
  )
  select * from log_feature

-- 数据库容器第一次启动时自动执行（只执行一次）。
-- POSTGRES_DB 只会建一个开发库 rag，测试库 rag_test 要在这里另外建
CREATE DATABASE rag_test;

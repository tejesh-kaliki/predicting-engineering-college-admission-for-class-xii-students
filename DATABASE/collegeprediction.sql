/*
SQLyog Community Edition- MySQL GUI v6.07
Host - 5.5.52-log : Database - collegeprediction
*********************************************************************
Server version : 5.5.52-log
*/

/*!40101 SET NAMES utf8 */;

/*!40101 SET SQL_MODE=''*/;

create database if not exists `collegeprediction`;

USE `collegeprediction`;

/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;

/*Table structure for table `signup` */

DROP TABLE IF EXISTS `signup`;

CREATE TABLE `signup` (
  `username` varchar(50) NOT NULL,
  `password` varchar(50) DEFAULT NULL,
  `contact_no` varchar(15) DEFAULT NULL,
  `gender` varchar(20) DEFAULT NULL,
  `email` varchar(50) DEFAULT NULL,
  `address` varchar(50) DEFAULT NULL,
  PRIMARY KEY (`username`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

/*Data for the table `signup` */

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;

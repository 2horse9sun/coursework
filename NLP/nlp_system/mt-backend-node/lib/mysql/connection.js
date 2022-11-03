let mysql = require('mysql');

let CONFIG = require('../../config')
//  生成连接池
//  CONFIG（配置）是全局变量
let pool = mysql.createPool(CONFIG.mysql);

//  封装异步操作
let query = function (sql, values) {
    return new Promise((resolve, reject) => {
        pool.query(sql, values, (err, rows) => {
            if (err) {
                console.log(err)
                reject(err)
            } else {
                console.log(rows)
                resolve(rows)
            }
        })
    })
};

module.exports = query;

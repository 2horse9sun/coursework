//人员管理mysql请求函数
let query = require('./connection');
var moment = require('moment')
let comment = require('./comment')

exports.get_random_sentence = async function(){
    let sql = "SELECT * FROM datasets WHERE id >= ((SELECT MAX(id) FROM datasets)-(SELECT MIN(id) FROM datasets)) * RAND() + (SELECT MIN(id) FROM datasets) LIMIT 1"
    let res = await query(sql)
    return res
}

exports.getlist = async function(zhi){
    let sql = "SELECT * FROM fxl5 where hot_word = ?"
    let reault = await query(sql,[zhi])
    return reault;
}

exports.getlistnice = async function(zhi){
    let sql = "SELECT * FROM niceword where word = ?"
    let reault = await query(sql,[zhi])
    return reault;
}

exports.getlength = async function(){
	let sql = "SELECT count(*) from fxl5";
	return reault = await query(sql,[])
}



exports.saveUserInfo = async function(userInfo,session_key,skey){
    let create_time = moment().format('YYYY-MM-DD HH:mm:ss');
    let update_time = create_time;
    let openId = userInfo.openId;
    let nickName = userInfo.nickName;
    let gender = userInfo.gender;
    let language = userInfo.language;
    let city = userInfo.city;
    let province = userInfo.province;
    let country = userInfo.country;
    let avatarUrl = userInfo.avatarUrl;

    let isuser = await comment.isuser(openId)
    if(isuser){
        console.log('已有用户')
        try{
            await comment.updateuser(openId,session_key,skey,update_time)
            let userInfo = await comment.getuserinfo(openId)
            return{
                reault: 0,
                skey: skey,
                userInfo: userInfo
            }
        }
        catch(e){
            return{
                reault: -4,
                errmsg: JSON.stringify(e)
            }
        }
        
    }
    else{
        console.log('新建用户')
        try{ 
            await comment.insertuser(openId,session_key,skey,nickName,gender,language,city,province,country,avatarUrl,create_time,update_time)
            let userInfo = await comment.getuserinfo(openId)
            return{
                reault: 0,
                skey: skey,
                userInfo: userInfo
            }
        }
        catch(e){
            return{
                reault: -4,
                errmsg: JSON.stringify(e)
            }
        }
        
    }

}

exports.addwenzi = async function(name,type,openid,content){
    let showtime = moment().format('YYYY-MM-DD HH:mm:ss');
    let sql = "insert into `dynamic` (`name`,`type`,`upload_openid`,`content`,`showtime`) values (?,?,?,?,?) ";
    await query(sql,[name,type,openid,content,showtime])
}

exports.addaudio = async function(name,type,openid,audio){
    let showtime = moment().format('YYYY-MM-DD HH:mm:ss');
    let sql = "insert into `dynamic` (`name`,`type`,`upload_openid`,`audio`,`showtime`) values (?,?,?,?,?) ";
    console.log([name,type,openid,audio,showtime])
    await query(sql,[name,type,openid,audio,showtime])
}

exports.uploadimg = async function(type,name,url,openid,txt){
    let showtime = moment().format('YYYY-MM-DD HH:mm:ss');
    let sql = "insert into `dynamic` (`name`,`type`,`upload_openid`,`content`,`showtime`,`img`) values (?,?,?,?,?,?) ";
    await query(sql,[name,type,openid,txt,showtime,url])
}


exports.uploadvideo = async function(type,name,url,openid,txt){
    let showtime = moment().format('YYYY-MM-DD HH:mm:ss');
    let sql = "insert into `dynamic` (`name`,`type`,`upload_openid`,`content`,`showtime`,`video`) values (?,?,?,?,?,?) ";
    await query(sql,[name,type,openid,txt,showtime,url])
}

exports.getranktype = async function(openid,type){
    let sql="select *,(select count(*) from dy_zan where dy_zan.dy_id = dynamic.id ) as app_num , (select count(*) from dy_zan where dy_zan.dy_id = dynamic.id and dy_zan.openid = ?) as iszan from dynamic,account where type = ? and account.openId = dynamic.upload_openid and dynamic.upload_openid = ? order by createtime desc";
    return await query(sql,[openid,type, openid])
}

exports.add_zan = async function(openid,dy_id){
    let showtime = moment().format('YYYY-MM-DD HH:mm:ss');
    let sql = "insert into dy_zan (`dy_id`,`openid`,`show_time`) values (?,?,?) "
    await query(sql,[dy_id,openid,showtime])
}

exports.del_zan = async function(openid,dy_id){
    let sql = "delete from dy_zan where openid = ? and dy_id = ?"
    await query(sql,[openid,dy_id])
}

exports.add_comment = async function(dy_id,type,comment_father_id,comment_beidong_id,content,people_zhudong,people_beidong){
    let showtime = moment().format('YYYY-MM-DD HH:mm:ss');
    let sql = "insert into dy_comment (`dy_id`,`type`,`comment_father_id`,`comment_beidong_id`,`content`,`people_zhudong`,`people_beidong`,`showtime`) values (?,?,?,?,?,?,?,?) "
    await query(sql,[dy_id,type,comment_father_id,comment_beidong_id,content,people_zhudong,people_beidong,showtime])
}

exports.del_comment = async function(comment_id){
    let sql = "delete from dy_comment where comment_id = ?"
    await query(sql,[comment_id])
}

exports.my_zan = async function(openid){
    let sql = "select *,(select avatarUrl from account where account.openId = ?) as touxiang, (select nickName from account where account.openId = ?) as ningzi from dy_zan,dynamic where dy_zan.openid = ? and dy_zan.dy_id = dynamic.id"
    return await query(sql,[openid,openid,openid])
}

exports.zan_my = async function(openid){
    let sql = "select * from dy_zan,dynamic,account where dy_zan.dy_id = dynamic.id and dynamic.upload_openid = ? and dy_zan.openid = account.openId"
    return await query(sql,[openid])
}

exports.my_comment = async function(openid){
    let sql = "select * from dy_dynamic where openid = ? order by createtime desc"
    return await query(sql,[openid])
}

exports.comment_my = async function(openid){

}

exports.getmydynamic = async function(openid,type){
    let sql = "select * from dynamic,account where dynamic.`upload_openid` = ? and dynamic.type = ? and dynamic.`upload_openid` = account.openId order by `createtime` desc"
    return await query(sql,[openid,type])
}

exports.deletemydynamic = async function(openid,type,dy_id){
    let sql = "delete from dynamic where `upload_openid` = ? and `type` = ? and `id` = ?";
    return await query(sql,[openid,type,dy_id])
}

exports.gettimelinedynamic = async function(openid){
    let sql = "select *, (select count(*) from dy_zan where dy_zan.dy_id = dynamic.id ) as app_num , (select count(*) from dy_zan where dy_zan.dy_id = dynamic.id and dy_zan.openid = ?) as iszan from dynamic,account where dynamic.`upload_openid` = account.openId order by createtime desc"
    return await query(sql,[openid])
}

exports.center_info = async function(openid,type){
    let sql = "select count(*) from dynamic where `upload_openid` = ? and type = ?"
    return await query(sql,[openid,type])
}
// exports.insertwebkey = async function(webkey){
//     let sql = "insert into `webkeylist` (`webkey`) values (?) ";
//     await query(sql,[webkey])
// }

// exports.webkeytimeout = async function(webkey){
//     let sql = "update `webkeylist` set `istime` = 1 where `webkey` = ? ";
//     await query(sql,[webkey])
// }

// exports.updatewebkey_wait = async function(openid,webkey,key){
//     let sql = "update `webkeylist` set `openid` = ?, `key` = ? where `webkey` = ? and `key` = 0";
//     await query(sql,[openid,key,webkey])
// }

// exports.updatewebkey = async function(openid,webkey,key){
//     let sql = "update `webkeylist` set `key` = ? where `openid` = ? and `webkey` = ?"
//     await query(sql,[key,openid,webkey])
// }


// exports.getwebkey = async function(webkey){
//     let sql = "select * from webkeylist where  `webkey` = ?"
//     return reault = await query(sql,[webkey])
// }

// exports.getopenidlist = async function(webkey){
//     let sql = "select * from webkeylist,account where  `webkey` = ? and webkeylist.openid = account.openid"
//     return reault = await query(sql,[webkey])
// }



// exports.listtop = async function(place){
//     let sql = "select * from fxl3 where hot_word = ?"
//     return reault = await query(sql,[place])
// }


// exports.getrandword = async function(num){
//     let sql = " select `word`,`id` from niceword order by rand() limit ?"
//     return reault = await query(sql,[num])
// }






// exports.getnew_img = async function(hot_word){
//     let sql ="select * from new_img where hot_word = ?";
//     return reault = await query(sql,[hot_word   ])
// }




// exports.getnewaccount = async function(account) {
//     let sql = "select * from newlist where account = ? order by rand() limit 3"
//     return reault = await query(sql,[account])
// }


// exports.getwordaccount = async function(account) {
//     let sql = "select * from wordlist where account = ? order by rand() limit 4 "
//     return reault = await query(sql,[account])
// }

// exports.getscaccount = async function(account){
//      let sql = "select * from shoucang where account = ? order by rand() limit 6 "
//     return reault = await query(sql,[account])
// }


// exports.insertwordlist = async function(zhi,account) {
//     let sql = "insert into `wordlist` (`word`,`account`) values (?,?)"
//     await query(sql,[zhi,account])
// }


// exports.isusersercert2 = async function(account) {
//     let sql = "select * from account2 where account = ?"
//     let reault = await query(sql,[account])
//     return reault.length
// } 

// exports.isusersercert = async function(account,sercret) {
//     let sql = "select * from account2 where account = ? and sercret = ?"
//     let reault = await query(sql,[account,sercret])
//     return reault.length
// } 

// exports.insertusersercert = async function(account,sercret){
//     let sql =" insert into account2 (`account`,`sercret`) values (?,?)"
//     await query(sql,[account,sercret])
// }








// exports.addnew = async function(account,url,title){
//     let sql =" insert into newlist (`account`,`url`,`title`) values (?,?,?)"
//     await query(sql,[account,url,title])
// }

// exports.addsc = async function(account,word){
//     let sql =" insert into shoucang (`account`,`word`) values (?,?)"
//     await query(sql,[account,word])
// }





// exports.getnicenews = async function(){
//     let sql ="select * from nicenews order by rand() limit 4"
//     return reault = await query(sql,[])
// }




// exports.num1 = async function(account){
//     let sql = "select * from wordlist where account = ?"
//     return reault = await query(sql,[account])
// }

// exports.num2 = async function(account){
//     let sql = "select * from newlist where account = ?"
//     return reault = await query(sql,[account])
// }

// exports.num3 = async function(account){
//     let sql = "select * from shoucang where account = ?"
//     return reault = await query(sql,[account])
// }

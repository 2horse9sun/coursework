let moment=require('moment');
module.exports = {
    defaultLayout: 'main',
    helpers: {
        if_eq: function (a, b, options) {
            return a != b ? options.fn(this) : options.inverse(this);
        },
        equal: function (a, b, options) {
            return a == b ? options.fn(this) : options.inverse(this);
        },
        notEqual: function (a, b, options) {
            return a === b ? options.inverse(this) : options.fn(this);
        },
        formatTime:function (time,options) {
            return moment(time).format('YYYY-MM-DD HH:mm:ss');
        }
        /*config: function (content, options) {
            if (config.pages[content]) {
                return options.fn(this);
            } else {
                return options.inverse(this);
            }
        }*/
    }
};
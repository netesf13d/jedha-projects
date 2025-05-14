///// This file contains sample NoSQL queries /////


// Get the user sessions with iOS as the operating system

db.session_data.find({
  "device_info.os": "iOS"
});


// Find all error log events that occured during the last month, last events first

const thresholdDate = new Date();
thresholdDate.setMonth(thresholdDate.getMonth() - 1);

db.logs.find({
  status: "error",
  timestamp: { $gte: thresholdDate }
}).sort({ timestamp: -1 });
// var cars = ["Saab", "Volvo", "BMW"];

// let text = "";

// for (let i = 0; i < cars.length; i++) {
//     text += '<p>' + cars[i] 
// }


var myVar = "test";

 $.ajax({
  url: "test.php",
  type: "POST",
  data:{"myData":myVar}
}).done(function(data) {
     console.log(data);
});

// console.log(cars.length);

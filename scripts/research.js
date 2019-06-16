$(document).ready(function(){
    $(".readmore").click(function(){
      $(this).next('.toggleblurb').slideToggle();
    });
  });
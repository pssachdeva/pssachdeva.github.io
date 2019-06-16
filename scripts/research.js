$().ready(function(){
    $(".toggleblurb").hide();
    $(".readmore").click(function(){
      $(this).next('.toggleblurb').slideToggle();
    });
  });
$().ready(function(){
    $(".toggleblurb").hide();
    $(".readmore").click(function(){
        debugger
      $(this).next('.toggleblurb').slideToggle();
    });
  });
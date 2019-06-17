$().ready(function(){
    $(".toggleblurb").hide();
    $(".readmore").click(function(){
      $(this).parent().parent().parent().find('.toggleblurb').slideToggle();
    });
  });
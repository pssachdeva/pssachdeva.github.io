$().ready(function(){
    $('.toggleblurb').hide();
    $(".readmore").click(function(){
      $('.toggleblurb').slideToggle();
    });
  });
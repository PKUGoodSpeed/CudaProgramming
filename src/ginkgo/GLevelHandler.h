#ifndef GLevelHandler_h
#define GLevelHandler_h
#include "../include/lglist.h"
#include "GOrder.h"
#include "GLevelOrderList.h"
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/transform.h>

namespace gpu_ginkgo{
    struct NotEqual{
        /* A comparitor for checking level book updates */
        __device__
        bool operator()(int x, int y) const{
            return x != y;
        }
    };
    
    
    /*
    Here we implement a Level handler struct for GPU.
    We use an array of pointers to handel levels.
    Pointers are pointing to a List of orders
    Each pointer handle one price level
    If there are no trade on that price level, the pointer is set NULL.
    */
    template<const int BASE_LEVEL = 1000, const int LEVEL_LIM = 300, const int LEVEL_ORDER_LIM = 5, const int MAX_POSITION = 20>
    struct LevelHandler{
        /*For later conveniece */
        typedef gpu_ginkgo::LevelOrderList<LEVEL_ORDER_LIM> gglol;
        typedef gpu_ginkgo::Order ggOrder;
        
        /* Aggregate information */
        int total_pnl, total_qty; 
        
        /* Position information */
        int pos, ppos;
        int max_pos;
        
        /* Book information */
        int ask_price, bid_price;
        int book_size[LEVEL_LIM];
        bool book_update[LEVEL_LIM];
        
        /* Strategy information */
        int ask_bbo, bid_bbo;
        float stgy_price, width;
        
        /* Strategy coefficients */
        float strong_ss_factor, weak_ss_factor;
        
        /* Array of Levels */
        gglol* price_levels[LEVEL_LIM];
        
        /* Time infomation */
        float cur_time, ltcy;
        
        /* min and max price level */
        int min_bl, max_bl;
        int min_sl, max_sl;
    
    public:
        /* Constructor */
        __device__ LevelHandler(const int&mpos, const float&sf, const float&wf):
        total_pnl(0), 
        total_qty(0),
        pos(0),
        ppos(0),
        max_pos(mpos),
        ask_price(0),
        bid_price(0),
        ask_bbo(0),
        bid_bbo(0),
        stgy_price(0),
        width(0),
        strong_ss_factor(sf),
        weak_ss_factor(wf),
        cur_time(0),
        ltcy(0),
        min_bl(LEVEL_LIM),
        max_bl(0),
        min_sl(LEVEL_LIM),
        max_sl(0) {
            memset(book_size, 0, sizeof(book_size));
            memset(price_levels, NULL, sizeof(price_levels));
            memset(book_update, false, sizeof(book_update));
        }
        
        /* Getting BookInfo */
        __device__ getBookInfo(int *bz, const int&ap, const int&bp, const float&smp, const float&t, const float &lag){
            thrust::transform(thrust::device, bz, bz + LEVEL_LIM, book_size, book_update, NotEqual());
            thrust::copy(thrust::device, bz, bz + LEVEL_LIM, book_size);
            ask_price = ap;
            bid_price = bp;
            stgy_price = smp;
            width = ask_price - bid_price;
            cur_time = t;
            ltcy = lag;
        }
        
        /* Get price from the level number */
        __device__ int getPrice(const int&l){
            return l + BASE_LEVEL;
        }
        
        /* Get level number from the price */
        __device__ int getLevel(const int&p){
            return p - BASE_LEVEL;
        }
        
        /* Computing ask bbo */
        __device__ void getAskBBO(){
            float tmp = -float(pos)/MAX_POSITION*(pos>=0? strong_ss_factor:weak_ss_factor) + stgy_price + 0.5*width;
            ask_bbo = int(tmp) + 1;
        }
        
        /* Computing bid bbo */
        __device__ void getBidBBO(){
            float tmp = -float(pos)/MAX_POSITION*(pos>=0? weak_ss_factor:strong_ss_factor) + stgy_price - 0.5*width;
            bid_bbo = int(tmp);
        }
        
        /* Adjust book */
        __device__ void adjustBook(){
            return ;
        }
        
        /* Process Trade */
        __device__ void ProcTrade(const bool&sell, int tv){
            if(sell){
                int j = getLevel(ask_price) - 1;
                while(j>=0 && !price_levels[j]) --j;
                if(getPrice(j) < bid_price) return;
                int fqty = 0, fpnl = 0, dq = 0, dqs = 0;
                price_levels[j]->procTrade(tv, fqty, fpnl, dq, dqs);
                if(price_levels[j].porders.empty() && price_levels[j]->orders.empty()){
                    delete price_levels[j];
                    price_levels[j] = NULL;
                }
                total_pnl += fpnl;
                total_qty += fqty;
                pos -= fqty;
            }
            else{
                int j = getLevel(bid_price) + 1;
                while(j<LEVEL_LIM && !price_levels[j]) ++j;
                if(getPrice(j) > ask_price) return;
                int fqty = 0, fpnl = 0, dq = 0, dqs = 0;
                price_levels[j]->procTrade(tv, fqty, fpnl, dq, dqs);
                if(price_levels[j].porders.empty() && price_levels[j]->orders.empty()){
                    delete price_levels[j];
                    price_levels[j] = NULL;
                }
                total_pnl -= fpnl;
                total_qty += fqty;
                pos += fqty;
            }
        }
        
        /* Swiping */
        __device__ void swipeLevel(const int&j){
            total_pnl += (sell? 1:-1) * price_levels[j]->level_qty * price_levels[j]->level_price;
            total_qty += price_levels[j]->level_qty;
            pos -= (sell? 1:-1) * price_levels[j]->level_qty;
            delete price_levels[j];
            price_levels[j] = NULL;
        }
        
        /* Sending New Order */
        __device__ void sendNewOrder(const int&j, const bool &sell, const int& qty){
            if(!price_levels[j]) price_levels = new gglol(getPrice(j), sell);
            if(!qty) return;
            price_levels[j].sendNewOrder(qty, cur_time + ltcy);
            ppos += (sell? -1:1)*qty;
        }
        
        /* Cancel Orders */
        __device__ void canclOrders(const int&j){
            if(cur_time > price_levels[j]->cancel_time){
                ppos += (price_levels[j]->sell? 1:-1)*price_levels[j]->level_qty;
                delete price_levels[j];
                price_levels[j] = NULL;
            }
        }
        
        /* Updating buying orders */
        __device__ void updateBuyOrders(){
            //1. Cancel latency passed
            for(int i=min_bl; i<=max_bl; ++i){
                if(price_levels[i]) canclOrders(i);
            }
            
            //2. Swipe fill
            for(int i = min(getLevel(ask_price), max_bl); i >= getLevel(ask_price); --i) {
                if(price_levels[i]) swipeLevel(i);
            }
            
            //3. Update q_value information
            for(int i=min_bl; i <= max_bl; ++i){
                if(price_levels[i]){
                    if(book_update[i]) price_levels[i].adjustQAgainstBU(book_size[i]);
                    price_levels[i].procPendingOrders(book_size[i], cur_time);
                }
            }
            
            //4. Update cancelation decision
            for(int i = max_bl; i > getLevel(bid_bbo); --i){
                if(price_levels[i] && !price_levels[i]->cancel) price_levels[i].preCancel(cur_time + ltcy);
            }
            for(int i = getLevel(bid_bbo)-5; i>=min_bl; --i){
                if(price_levels[i] && !price_levels[i]->cancel) price_levels[i].preCancel(cur_time + ltcy);
            }
        }
        
        /* Sending new buying level orders */
        __device__ void sendNewBuyOrders(){
            for(int i=getLevel(bid_bbo); i >= max(0, getLevel(bid_bbo)-4) && ppos<max_pos; --i){
                /*
                If this level has been determined to be canceled
                and holding uncanceld orders, we are not sending order
                at this level.
                */
                if(price_levels[i] && price_levels[i]->cancel) continue;
                /* Make sure some constraints are satisfied. */
                int tmp_qty = (price_levels[i]? LEVEL_ORDER_LIM-price_levels[i]->level_qty: LEVEL_ORDER_LIM);
                sendNewOrder(i, false, min(max_pos - ppos, tmp_qty));
            }
        }
        
        /* Updating buying level limits */
        __device__ void updateBuyLevelLimits(){
            max_bl = max(max_bl, getLevel(bid_bbo));
            while(max_bl>=0 && !price_levels[max_bl]) --max_bl;
            min_bl = min(min_bl, getLevel(bid_bbo)-4);
            while(min_bl<LEVEL_LIM && !price_levels[min_bl]) ++min_bl;
        }
        
        /* Updating selling orders */
        __device__ void updateBuyOrders(){
            //1. Cancel latency passed
            for(int i=min_sl; i<=max_sl; ++i){
                if(price_levels[i]) canclOrders(i);
            }
            
            //2. Swipe fill
            for(int i = max(getLevel(bid_price), min_sl); i <= getLevel(bid_price); ++i) {
                if(price_levels[i]) swipeLevel(i);
            }
            
            //3. Update q_value information
            for(int i=min_sl; i <= max_sl; ++i){
                if(price_levels[i]){
                    if(book_update[i]) price_levels[i].adjustQAgainstBU(book_size[i]);
                    price_levels[i].procPendingOrders(book_size[i], cur_time);
                }
            }
            
            //4. Update cancelation decision
            for(int i = min_sl; i < getLevel(ask_bbo); ++i){
                if(price_levels[i] && !price_levels[i]->cancel) price_levels[i].preCancel(cur_time + ltcy);
            }
            for(int i = getLevel(ask_bbo)+5; i<=max_sl; ++i){
                if(price_levels[i] && !price_levels[i]->cancel) price_levels[i].preCancel(cur_time + ltcy);
            }
        }
        
        /* Sending new selling level orders */
        __device__ void sendNewBuyOrders(){
            for(int i=getLevel(ask_bbo); i < min(LEVEL_LIM, getLevel(ask_bbo)+5) && ppos>-max_pos; ++i){
                /*
                If this level has been determined to be canceled
                and holding uncanceld orders, we are not sending order
                at this level.
                */
                if(price_levels[i] && price_levels[i]->cancel) continue;
                /* Make sure some constraints are satisfied. */
                int tmp_qty = (price_levels[i]? LEVEL_ORDER_LIM-price_levels[i]->level_qty: LEVEL_ORDER_LIM);
                sendNewOrder(i, true, min(max_pos + ppos, tmp_qty));
            }
        }
        
        /* Updating selling level limits */
        __device__ void updateBuyLevelLimits(){
            max_sl = max(max_sl, getLevel(ask_bbo)+4);
            while(max_sl>=0 && !price_levels[max_sl]) --max_sl;
            min_sl = min(min_sl, getLevel(ask_bbo));
            while(min_sl<LEVEL_LIM && !price_levels[min_sl]) ++min_sl;
        }
        
    };
}


#endif
#ifndef GOrderHandler_h
#define GOrderHandler_h
#include "GOrder.h"
#include "GOrderList.h"
#include "../include/lglist.h"
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <cassert>

namespace gpu_ginkgo{
    /*
    Here we implement an Order Handler struct which can be used in GPU.
    The Order Handler struct is used to process different types of orders.
    It mainly contain two GOrderList stuct, one for buying and one for selling
    */
    template<const int ORDER_LIM = 100, const int LEVEL_LIM = 200>
    struct OrderHandler{
        /* For Later convenience */
        typedef gpu_ginkgo::Order ggOrder;
        typedef gpu_ginkgo::OrderList<ORDER_LIM, LEVEL_LIM> ggOList;
        
        /* Main containers for orders */
        ggOList sell_list;
        ggOList buy_list;
        
        /* Aggregate information */
        int total_pnl, total_qty; 
        
        /* Position information */
        int pos;
        int max_pos;
        
        /* Book Information */
        int ask_price, bid_price;
        int book_sizes[LEVEL_LIM];
        
        /* Strategy information */
        int ask_bbo, bid_bbo;
        float stgy_price;
        
        /* Strategy coefficients */
        float strong_ss_factor, weak_ss_factor;
        
        /* Trade Information */
        bool trade_sell;
        int trade_price, trade_volume;
        
        /* Our order price lim information */
        int base_p;
        int min_sp;
        int max_bp;
    public:
        /* Constructor */
        __device__ OrderHandler(const int&base_price, const int&mlq):
        sell_list(true, base_price, mlq),
        buy_list(false, base_price, mlq),
        total_pnl(0),
        total_qty(0),
        max_pos(0),
        pos(0),
        ask_price(base_price + LEVEL_LIM),
        bid_price(base_price - 1),
        ask_bbo(0),
        bid_bbo(0),
        stgy_price(0),
        strong_ss_factor(0),
        weak_ss_factor(0),
        trade_sell(true),
        trade_price(0),
        trade_volume(0),
        base_p(base_price),
        min_sp(base_price + LEVEL_LIM),
        max_bp(base_price) {
            memset(book_sizes, 0, sizeof(book_sizes));
        }
        
        /* Load Strategy information*/
        __device__ void loadStrategy(const int&mpos, const float&sf, const float&wf){
            max_pos = mpos;
            strong_ss_factor = sf;
            weak_ss_factor = wf;
        }
        
        /* Getting time information */
        __device__ void getTimeInfo(const float&t, const float&ltcy){
            sell_list.getTime(t, ltcy);
            buy_list.getTime(t, ltcy);
        }
        
        /* Getting BookInfo */
        __device__ void getBookInfo(int *bz, const int&ap, const int&bp, const float&smp){
            thrust::copy(thrust::device, bz, bz + LEVEL_LIM, book_sizes);
            ask_price = ap;
            bid_price = bp;
            stgy_price = smp;
        }
        
        /* Computing ask bbo */
        __device__ void getAskBBO(){
            float width = ask_price - bid_price, ppos = pos + buy_list.side_qty - sell_list.side_qty;
            float tmp = -float(ppos)/max_pos*(ppos>=0? strong_ss_factor:weak_ss_factor) + stgy_price + 0.5*width;
            ask_bbo = ceil(tmp) + 1;
            ask_bbo = ask_price;
        }
        
        /* Computing bid bbo */
        __device__ void getBidBBO(){
            float width = ask_price - bid_price, ppos = pos + buy_list.side_qty - sell_list.side_qty;
            float tmp = -float(ppos)/max_pos*(ppos>=0? weak_ss_factor:strong_ss_factor) + stgy_price - 0.5*width;
            bid_bbo = floor(tmp);
            bid_bbo = bid_price;
        }
        
        /* Book Update simulation */
        __device__ void bookUpdateSim(int *bz, const int&ap, const int&bp, const float&smp){
            //Load book info
            this->getBookInfo(bz, ap, bp, smp);
            
            // Updating sell_list
            sell_list.updatePendingOrders(book_sizes);
            sell_list.updateCancelOrders(bid_price, book_sizes);
            sell_list.updateAckedOrders(bid_price, book_sizes);
            sell_list.reset(pos, total_pnl, total_qty);
            while(sell_list.levelEmpty(min_sp)) ++min_sp;
            
            // Updating buy_list
            buy_list.updatePendingOrders(book_sizes);
            buy_list.updateCancelOrders(ask_price, book_sizes);
            buy_list.updateAckedOrders(ask_price, book_sizes);
            buy_list.reset(pos, total_pnl, total_qty);
            while(buy_list.levelEmpty(max_bp)) --max_bp;
            
            // Assertion min_sp > max_bp; to avoid self trade
            assert(min_sp > max_bp);
        }
        
        /* One function to handle new orders*/
        __device__ void cancelAndSendNewOrders(){
            // Computing BBOs
            this->getAskBBO();
            this->getBidBBO();
            
            // Determine cancelation of sell_list
            // We are also avoiding cross spread trade
            int alp = max(ask_bbo, max(ask_price, max_bp + 1));
            int ahp = ask_bbo + 4;  // Up to 5 price levels
            sell_list.preCanceling(alp, ahp);
            
            // Determine cancelation of buy_list
            // We are also avoiding cross spread trade
            int bhp = min(bid_bbo, min(bid_price, min_sp - 1));
            int blp = bid_bbo - 4;  // Orders range up to 5 price levels
            buy_list.preCanceling(blp, bhp);
            
            
            //Sending new selling orders
            int qty_lim;
            qty_lim = max(0, max_pos + pos - sell_list.side_qty);
            sell_list.sendNewSellingOrders(alp, ahp, qty_lim);

            qty_lim = max(0, max_pos - pos - buy_list.side_qty);
            buy_list.sendNewBuyingOrders(blp, bhp, qty_lim);
            
            // Updating price limits
            min_sp = min(min_sp, alp);
            while(sell_list.levelEmpty(min_sp)) ++min_sp;
            max_bp = max(max_bp, bhp);
            while(buy_list.levelEmpty(max_bp)) --max_bp;
            
            // Assertion min_sp > max_bp; to avoid self trade
            assert(min_sp > max_bp);
        }
        
        /* Load Trade Information */
        __device__ void getTradeInfo(const bool&sell, const int &tp, const int &tv){
            trade_sell = sell;
            trade_price = tp;
            trade_volume = tv;
        }
        
        /* Process a sell or buy trade */
        __device__ void processTrade(const bool&sell, const int &tp, int &tv){
            // We need to make the following statement true when while doing presampling
            assert(sell_list.getLevel(tp) >= 0 && sell_list.getLevel(tp) < LEVEL_LIM);
            
            // Updating sell_list
            sell_list.updatePendingOrders(book_sizes);
            sell_list.updateCancelOrders(bid_price, book_sizes);
            sell_list.updateAckedOrders(bid_price, book_sizes);
            sell_list.reset(pos, total_pnl, total_qty);
            while(sell_list.levelEmpty(min_sp)) ++min_sp;
            
            // Updating buy_list
            buy_list.updatePendingOrders(book_sizes);
            buy_list.updateCancelOrders(ask_price, book_sizes);
            buy_list.updateAckedOrders(ask_price, book_sizes);
            buy_list.reset(pos, total_pnl, total_qty);
            while(buy_list.levelEmpty(max_bp)) --max_bp;
            
            if(sell){
                /*
                If the aggressive trade is sell, then the bid/buy_list side will
                react against it.
                */
                for(int prc = max(bid_price, max_bp); prc>=tp && tv; --prc){
                    int idx = buy_list.getLevel(prc);
                    buy_list.getTradedThrough(tv, prc, book_sizes);
                    int delta_v = min(book_sizes[idx], tv);
                    book_sizes[idx] -= delta_v;
                    tv -= delta_v;
                }
                
                // Updating position and pnl
                buy_list.reset(pos, total_pnl, total_qty);
                
                // Updating price limits
                while(buy_list.levelEmpty(max_bp)) --max_bp;
            }
            else{
                /*
                If the aggressive trade is buy, the the ask/sell_list side will
                react against it.
                */
                for(int prc = min(ask_price, min_sp); prc<=tp && tv; ++prc){
                    int idx = sell_list.getLevel(prc);
                    sell_list.getTradedThrough(tv, prc, book_sizes);
                    int delta_v = min(book_sizes[idx], tv);
                    book_sizes[idx] -= delta_v;
                    tv -= delta_v;
                }
                
                // Updating position and pnl
                sell_list.reset(pos, total_pnl, total_qty);
                
                // Updating price limits
                while(sell_list.levelEmpty(min_sp)) ++min_sp;
            }
            
            // Assertion min_sp > max_bp; to avoid self trade
            assert(min_sp > max_bp);
        }
        
        /* Show Basic Information */
        __device__ void showBasicInfo(){
            printf("\n======================= ORDER Info =======================\n");
            printf("Current Position = %d\n", pos);
            printf("Total Pnl = %d\n", total_pnl);
            printf("Total Trade qty = %d\n", total_qty);
            printf("\n========================== END ===========================\n");
        }
        
        /* Show Order Book Information */
        __device__ void showOrderBookInfo(){
            printf("\n======================= ORDER BOOK =======================\n");
            printf("\n<<<<<  SELLING ORDERS  >>>>>\n");
            sell_list.showLevelQtyInfo();
            sell_list.showPendingOrderInfo();
            sell_list.showAckedOrderInfo();
            sell_list.showCanceledOrderInfo();
            sell_list.showUpdateInfo();
            
            printf("\n<<<<<  BUYING ORDERS  >>>>>\n");
            buy_list.showLevelQtyInfo();
            buy_list.showPendingOrderInfo();
            buy_list.showAckedOrderInfo();
            buy_list.showCanceledOrderInfo();
            buy_list.showUpdateInfo();
            printf("\n========================== END ===========================\n");
        }
    };
}

#endif